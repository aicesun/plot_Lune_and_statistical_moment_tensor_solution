#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor inversion result (from MTFIT) and plot Lune with fitted gaussian and associated statistical unconstrained MT inversion plot.
# What the script does:
# For each event:
# 1. Takes specified sample of solutions (highest probability specificed fraction of MT solutions)
# 2. Plots Lune for these data
# 3. Fits 2D gaussian to Lune plot, extracting a peak location and a contour (representing the half width maximum, standard devation or similar)
# 4. Takes the peak location from the gaussian and plots a random sample of solutions from that.
# 5. Outputs MT beachball style plot with the potential solutions on, along with the radiation pattern for the most likely solution from the entire MT inversion for the event.

# Input variables:
# MT_inversion_result_mat_file_path - Path to full unconstrained moment tensor inversion result


# Output variables:
# Plots saved to file.

# Created by Tom Hudson, 10th January 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
from numpy.linalg import eigh # For calculating eigenvalues and eigenvectors of symetric (Hermitian) matrices
import matplotlib
import matplotlib.pyplot as plt
import obspy
import scipy.io as sio # For importing .mat MT solution data
import scipy.optimize as opt # For curve fitting
import math # For plotting contours as line
import os,sys
import random
from matplotlib import path # For getting circle bounding path for MT plotting
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall # For getting nodal planes for unconstrained moment tensors
from obspy.core.event.source import farfield # For calculating MT radiation patterns
from matplotlib.patches import Polygon, Circle # For plotting MT radiation patterns
from matplotlib.collections import PatchCollection # For plotting MT radiation patterns
import glob


# ------------------- Define generally useful functions -------------------
def load_MT_dict_from_file(matlab_data_filename):
    data=sio.loadmat(matlab_data_filename)
    i=0
    while True:
        try:
            # Load data UID from matlab file:
            if data['Events'][0].dtype.descr[i][0] == 'UID':
                uid=data['Events'][0][0][i][0]
            if data['Events'][0].dtype.descr[i][0] == 'Probability':
                MTp=data['Events'][0][0][i][0] # stored as a n length vector, the probability
            if data['Events'][0].dtype.descr[i][0] == 'MTSpace':
                MTs=data['Events'][0][0][i] # stored as a 6 by n array (6 as 6 moment tensor components)
            i+=1
        except IndexError:
            break
    
    try:
        stations = data['Stations']
    except KeyError:
        stations = []
        
    return uid, MTp, MTs, stations


def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    

def get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample):
    """Function to return fraction of MTs based on highet probabilities. Also returns associated probabilities."""
    num_events_to_sample = int(len(MTp)*frac_to_sample) # Take top 1 % of samples
    sorted_indices = np.argsort(MTp)[::-1] # reorder into descending order
    # Find indices of solutions in sample:
    sample_indices = sorted_indices[0:num_events_to_sample]
    MTs_sample = MTs[:,sample_indices]
    MTp_sample = MTp[sample_indices]
    print "Sampled",len(MTs_sample[0,:]),"out of",len(MTs[0,:]),"events"
    return MTs_sample, MTp_sample
    

def find_delta_gamm_values_from_sixMT(sixMT):
    """Function to find delta and gamma given 6 moment tensor."""
    # Get full MT:
    MT_current = sixMT
    # And get full MT matrix:
    full_MT_current = get_full_MT_array(MT_current)

    # Find the eigenvalues for the MT solution and sort into descending order:
    w,v = eigh(full_MT_current) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    full_MT_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order

    # Calculate gamma and delta (lat and lon) from the eigenvalues:
    lambda1 = full_MT_eigvals_sorted[0]
    lambda2 = full_MT_eigvals_sorted[1]
    lambda3 = full_MT_eigvals_sorted[2]
    # print (lambda1**2 + lambda2**2 + lambda3**2)**0.5 # Should = 1 if normallised correctly
    gamma = np.arctan(((-1*lambda1) + (2*lambda2) - lambda3)/((3**0.5)*(lambda1 - lambda3))) # eq. 20a (Tape and Tape 2012)
    beta = np.arccos((lambda1+lambda2+lambda3)/((3**0.5)*((lambda1**2 + lambda2**2 + lambda3**2)**0.5))) # eq. 20b (Tape and Tape 2012)
    delta = (np.pi/2.) - beta # eq. 23 (Tape and Tape 2012)

    return delta, gamma


def equal_angle_stereographic_projection_conv_YZ_plane(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    Y = y/(1+x)
    Z = z/(1+x)
    return Y,Z
    

def threeD_spher_coords_to_twoD_Lune_proj_conversion(r,theta,phi):
    """Function to convert 3D spherical coords into 2D Lune projection coords (X,Z plane). (Theta 0 -> pi, phi 0 -> 2pi)"""
    X = r*np.sin(theta)*np.sin(phi)
    Z = r*np.cos(theta)
    return X,Z
    

def convert_spherical_coords_to_cartesian_coords(r,theta,phi):
    """Function to take spherical coords and convert to cartesian coords. (theta between 0 and pi, phi between 0 and 2pi)"""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z
    

def twoD_Gaussian((X, Y), amplitude, xo, yo, sigma_x, sigma_y, theta):
    """Function describing 2D Gaussian. Pass initial guesses for gaussian parameters. Returns 1D ravelled array describing 2D Gaussian function.
    Based on code: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    X, Y are 2D np grids (from np.meshgrid)."""
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gau = amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) + c*((Y-yo)**2)))
    #gau_out = gau.ravel() # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    gau_out = np.ravel(gau) # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    return gau_out


def fit_twoD_Gaussian(x, y, data, initial_guess_switch=False, initial_guess=(1,1,1,1,1)):
    """Function to fit 2D Gaussian to a dataset. x, y are 1D data arrays, data is a 2D array, described by x and y as labels.
    Based on code from:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m"""    
    
    # Mesh grid for 2D Gaussian fit:
    Y, X = np.meshgrid(y, x)
    
    # Fit Gaussian to data:
    data_ravelled = np.ravel(data)
    if initial_guess_switch:
        print "Initial guess parameters for 2D gaussian fit:"
        print initial_guess
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled, p0=initial_guess)
    else:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled)
    print "And final parameters derived:"
    print popt
    
    # Get fitted data:
    data_fitted = twoD_Gaussian((X, Y), *popt) # Get 2D Gaussian
    data_fitted = np.reshape(data_fitted, np.shape(data)) # and reshape to original data dimensions
    
    return data_fitted
    

# Get MT solutions binned and stored for all gamma and delta values:
def get_binned_MT_solutions_by_delta_gamma_dict(MTs, MTp, return_all_switch=False):
    """Function to get binned MT solutions by delta and gamma value. Input is array of MTs (in (6,n) shape) and array of associated probabilities (MTp, length n).
    Output is binned dictionary containing bin values of delta and gamma and all MT solutions that are in the bin."""
    
    # Set up store for binned MT data:
    gamma_delta_binned_MT_store = {} # Will have the entries: gamma_delta_binned_MT_store[delta][gamma]: [MTs(shape(6,n))] and [MTp (shape n)]

    # Setup delta-gamma bins for data:
    bin_size_delta = np.pi/120. #np.pi/60.
    bin_size_gamma = np.pi/120. #np.pi/60.
    bin_value_labels_delta = np.arange(-np.pi/2,np.pi/2+bin_size_delta, bin_size_delta)
    bin_value_labels_gamma = np.arange(-np.pi/6,np.pi/6+bin_size_gamma, bin_size_gamma)
    bins_delta_gamma = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float) # array to store bin values (although can also obtain from dictionary sizes)
    bins_delta_gamma_probability_array = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float) # array to store probability associated with each bin (although can also obtain from dictionary sizes)
    
    # And setup dict for all binned values:
    for delta in bin_value_labels_delta:
        for gamma in bin_value_labels_gamma:
            try:
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
            except KeyError:
                gamma_delta_binned_MT_store["delta="+str(delta)] = {}
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
    
    # Loop over events (binning each data point):
    for a in range(len(MTs[0,:])):
        # Get delta and gamma values for sixMT:
        MT_current = MTs[:,a]
        MTp_current = MTp[a]
        delta, gamma = find_delta_gamm_values_from_sixMT(MT_current)

        # And bin solution into approriate bin:
        idx_delta = (np.abs(bin_value_labels_delta-delta)).argmin()
        idx_gamma = (np.abs(bin_value_labels_gamma-gamma)).argmin()
        bins_delta_gamma[idx_delta,idx_gamma] += 1. # Append 1 to bin
        bins_delta_gamma_probability_array[idx_delta,idx_gamma] += MTp_current # Sum probability for given bin
        
        # And add to dictionary:
        delta_bin_label_tmp = bin_value_labels_delta[idx_delta]
        gamma_bin_label_tmp = bin_value_labels_gamma[idx_gamma]
        try:
            tmp_MT_stacked_array = gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"]
            tmp_MTp_array = gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTp"]
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.hstack((tmp_MT_stacked_array, MT_current.reshape(6,1)))
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTp"] = np.hstack((tmp_MTp_array, np.array([MTp_current])))
        except KeyError:
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.array(MT_current.reshape(6,1)) # If doesnt exist, create new MT store entry
            gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTp"] = np.array([MTp_current]) # If doesnt exist, create new MTp store entry
    
    if return_all_switch:
        return gamma_delta_binned_MT_store, bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, bins_delta_gamma_probability_array
    else:
        return gamma_delta_binned_MT_store


# Define useful functions for plotting MTs:
def get_nodal_plane_xyz_coords(mt_in):
    """Function to get nodal plane coords given 6 MT in, in NED coords. Returns 2 arrays, describing the two nodal planes in terms of x,y,z coords on a unit sphere."""
    ned_mt = mt_in # 6 MT
    mopad_mt = MomentTensor(ned_mt,system='NED') # In north, east, down notation
    bb = BeachBall(mopad_mt, npoints=200)
    bb._setup_BB(unit_circle=True)
    neg_nodalline = bb._nodalline_negative # extract negative nodal plane coords (in 3D x,y,z)
    pos_nodalline = bb._nodalline_positive # extract positive nodal plane coords (in 3D x,y,z)
    return neg_nodalline, pos_nodalline


def equal_angle_stereographic_projection_conv_XY_plane(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    X = x/(1+z)
    Y = y/(1+z)
    return X,Y


def Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    X = x * np.sqrt(2/(1+z))
    Y = y * np.sqrt(2/(1+z))
    return X,Y


# ------------------- End of defining generally useful functions -------------------


# ------------------- Define significant functions -------------------

def plot_sampled_MT_solns_on_Lune_with_gaussian_fit(MTs, MTp, frac_to_sample=0.1, figure_filename=[]):
    """Function to plot sampled MT solutions on Lune, binned. Will also fit gaussian to this and return the maximum location of the gaussian and the contour coordinates. Also outputs saved figure."""
    
    # Get sample of MT solutions for fitting Gaussian to:
    MTs_sample, MTp_sample = get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample)
    
    # Get bin values and bin probability values for delta-gamma space (for plotting Lune):
    gamma_delta_binned_MT_store, bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, bins_delta_gamma_probability_array = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample, MTp_sample, return_all_switch=True)

    # Fit 2D gaussian to delta-gamma Lune data:
    use_bin_values_vs_prob_values_switch = "prob" # "bin"
    # Define initial guess params:
    if use_bin_values_vs_prob_values_switch == "bin":
        amplitude = np.max(bins_delta_gamma)
    elif use_bin_values_vs_prob_values_switch == "prob":
        amplitude = np.max(bins_delta_gamma_probability_array)
    xo = 0.0
    yo = 0.0
    sigma_x = np.pi/6.
    sigma_y = np.pi/8.
    theta = np.pi/2.
    initial_guess=(amplitude, xo, yo, sigma_x, sigma_y, theta) # Define initial guess values from data
    # And fit gaussian:
    if use_bin_values_vs_prob_values_switch == "bin":
        bins_delta_gamma_gau_fitted = fit_twoD_Gaussian(bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, initial_guess_switch=True, initial_guess=initial_guess)
        print "Lune plot setting: binned number of solutions per delta,gamma"
    elif use_bin_values_vs_prob_values_switch == "prob":
        bins_delta_gamma_gau_fitted = fit_twoD_Gaussian(bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma_probability_array, initial_guess_switch=True, initial_guess=initial_guess)
        print "Lune plot setting: cummulative probability"
    
    # Get location of maximum of Gaussian fit and 1 stdev contour:
    # Get location of maximum:
    max_bin_delta_gamma_indices = np.where(bins_delta_gamma_gau_fitted==np.max(bins_delta_gamma_gau_fitted))
    max_bin_delta_gamma_values = [bin_value_labels_delta[max_bin_delta_gamma_indices[0][0]], bin_value_labels_gamma[max_bin_delta_gamma_indices[1][0]]]
    # Get contour:
    contour_val = bins_delta_gamma_gau_fitted[max_bin_delta_gamma_indices[0][0], max_bin_delta_gamma_indices[1][0]]/2. #np.std(bins_delta_gamma_gau_fitted)
    plus_minus_range = 0.05
    contour_delta_values_indices = []
    contour_gamma_values_indices = []
    contour_delta_values = []
    contour_gamma_values = []
    for i in range(len(bins_delta_gamma_gau_fitted[:,0])):
        for j in range(len(bins_delta_gamma_gau_fitted[0,:])):
            if bins_delta_gamma_gau_fitted[i,j]<=contour_val*(1.+plus_minus_range) and bins_delta_gamma_gau_fitted[i,j]>=contour_val*(1.-plus_minus_range):
                # Find contour values:
                contour_delta_values_indices.append(i)
                contour_gamma_values_indices.append(j)
                contour_delta_values.append(bin_value_labels_delta[i])
                contour_gamma_values.append(bin_value_labels_gamma[j])        
    # And sort contour points into clockwise order:
    pts = []
    for i in range(len(contour_delta_values)):
        pts.append([contour_delta_values[i], contour_gamma_values[i]])
    try:
        origin = pts[0]
    except:
        print "Not enough points associated with contour therefore increase value of plus_minus_range variable in function plot_sampled_MT_solns_on_Lune_with_gaussian_fit()"
        sys.exit()
    refvec = [0, 1]
    # Define function for plotting contour:
    def clockwiseangle_and_distance(point):
        """Function to order points in clockwise order. Needs origin and refvec defined.
        Code from: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python"""
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    contour_bin_delta_gamma_values_sorted = sorted(pts, key=clockwiseangle_and_distance) # Sorts points into clockwise order
    contour_bin_delta_gamma_values_sorted.append(contour_bin_delta_gamma_values_sorted[0]) # Append first point again to make circle
    
    # And plot:
    print "Plotting Lune with fitted Gaussian"
    # Set up figure:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    # Plot major gridlines:
    for phi in [-np.pi/6., np.pi/6.]:
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black")
    # Plot horizontal minor grid lines:
    minor_horiz_interval = np.pi/12.
    for theta in np.arange(0.+minor_horiz_interval, np.pi+minor_horiz_interval, minor_horiz_interval):
        phi_range = np.linspace(-np.pi/6,np.pi/6,90)
        theta_range = np.ones(len(phi_range))*theta
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)
    # Plot vertical minor gridlines:
    minor_vert_interval = np.pi/24.
    for phi in np.arange(-np.pi/6+minor_vert_interval, np.pi/6, minor_vert_interval):
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)

    # And plot binned data, colored by bin value:
    bins_delta_gamma_normallised = bins_delta_gamma/np.max(bins_delta_gamma) # Normalise data
    # Loop over binned data points:
    for i in range(len(bin_value_labels_delta)):
        for j in range(len(bin_value_labels_gamma)):
            delta = bin_value_labels_delta[i]
            gamma = bin_value_labels_gamma[j]
            # And plot data coord:
            x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
            Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
            ax.scatter(Y,Z, color = matplotlib.cm.jet(int(bins_delta_gamma_normallised[i,j]*256)), alpha=0.6,s=50)
        print i

    # Plot maximum location and associated contours associated with Guassian fit:
    # Plot maximum location:
    delta = max_bin_delta_gamma_values[0]
    gamma = max_bin_delta_gamma_values[1]
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    ax.scatter(Y,Z, color = "green", alpha=1.0,s=50, marker="X")
    # And plot 1 stdev contour:
    contour_bin_delta_values_sorted = []
    contour_bin_gamma_values_sorted = []
    for i in range(len(contour_bin_delta_gamma_values_sorted)):
        contour_bin_delta_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][0])
        contour_bin_gamma_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][1])
    delta = np.array(contour_bin_delta_values_sorted)
    gamma = np.array(contour_bin_gamma_values_sorted)
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    ax.plot(Y,Z, color = "green", alpha=0.5)

    # And Finish plot:
    # Plot labels for various defined locations (locations from Tape and Tape 2012, table 1):
    plt.scatter(0.,1.,s=50,color="black")
    plt.text(0.,1.,"Explosion", fontsize=12, horizontalalignment="center", verticalalignment='bottom')
    plt.scatter(0.,-1.,s=50,color="black")
    plt.text(0.,-1.,"Implosion", fontsize=12, horizontalalignment="center", verticalalignment='top')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - np.arcsin(5/np.sqrt(33)),-np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^+$",color="red", fontsize=12, horizontalalignment="right", verticalalignment='bottom')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) + np.arcsin(5/np.sqrt(33)),np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^-$",color="red", fontsize=12, horizontalalignment="left", verticalalignment='top')
    plt.scatter(0.,0.,s=50,color="red")
    plt.text(0.,0.,"DC",color="red", fontsize=12, horizontalalignment="center", verticalalignment='top')
    # Various tidying:
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
    plt.axis('off')
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
    
    # And return MT data at maximum (and mts within contour?!):
    print "And getting MT data at maximum of gaussian to return (and mts within contour?!)"
    # Get all solutions associated with bins inside contour on Lune plot:
    gamma_delta_binned_MT_store = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample, MTp_sample) # Returns dictionary of all MTs binned by gamma, delta value
    # And get all values associated with gaussian maximum on Lune plot:
    max_bin_delta_gamma_indices = np.where(bins_delta_gamma_gau_fitted==np.max(bins_delta_gamma_gau_fitted))
    max_bin_delta_gamma_values = [bin_value_labels_delta[max_bin_delta_gamma_indices[0][0]], bin_value_labels_gamma[max_bin_delta_gamma_indices[1][0]]]
    delta = max_bin_delta_gamma_values[0]
    gamma = max_bin_delta_gamma_values[1]
    try:
        MTs_max_gau_loc = gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)]["MTs"] # MT solutions associated with gaussian maximum (note: may be different to maximum value due to max value being fit rather than real value)
    except KeyError:
        print "Insufficient sample of MTs to produce a solution. Try again with greater fraction."
        sys.exit()
    return MTs_max_gau_loc


def plot_MTs_on_sphere_twoD(MTs_to_plot, radiation_pattern_MT=[], stations=[], radiation_MT_phase = "P", lower_upper_hemi_switch="lower", figure_filename=[], num_MT_solutions_to_plot=20):
    """Function to plot MT solutions on sphere, then project into 2D using an equal area projection.
    Input MTs are np array of NED MTs in shape [6,n] where n is number of solutions. Also takes optional radiation_pattern_MT, which it will plot a radiation pattern for.
    Note: x and y coordinates switched for plotting to take from NE to EN
    Note: stations is a dictionary containing station info."""
    
    print "Plotting MT solutions on beachball for:"
    print "For", len(MTs_to_plot[0,:]), "solutions."
    if not len(radiation_pattern_MT)==0:
        print "And radiation pattern for specified MT."
    
    # Setup figure:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111) #projection="3d")
    
    # Setup bounding circle:
    theta = np.ones(200)*np.pi/2
    phi = np.linspace(0.,2*np.pi,len(theta))
    r = np.ones(len(theta))
    x,y,z = convert_spherical_coords_to_cartesian_coords(r,theta,phi)
    X_bounding_circle,Y_bounding_circle = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    ax.plot(Y_bounding_circle,X_bounding_circle)
    
    # And create bounding path from circle:
    path_coords = [] # list to store path coords
    for i in range(len(X_bounding_circle)):
        x_tmp = X_bounding_circle[i]
        y_tmp = Y_bounding_circle[i]
        path_coords.append((x_tmp,y_tmp))
    bounding_circle_path = path.Path(path_coords) # bounding path that can be used to find points
    #bounding_circle_path.contains_points([(.5, .5)])
    
    # Plot radiation pattern if provided with radiation pattern MT to plot:
    if not len(radiation_pattern_MT)==0:
        # Get MT to plot radiation pattern for:
        ned_mt = radiation_pattern_MT

        # Get spherical points to sample for radiation pattern:
        theta = np.linspace(0,np.pi,100)
        phi = np.linspace(0.,2*np.pi,len(theta))
        r = np.ones(len(theta))
        THETA,PHI = np.meshgrid(theta, phi)
        theta_flattened = THETA.flatten()
        phi_flattened = PHI.flatten()
        r_flattened = np.ones(len(theta_flattened))
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_flattened,theta_flattened,phi_flattened)
        if lower_upper_hemi_switch=="upper":
            z = -1*z
        radiation_field_sample_pts = np.vstack((x,y,z))
        # get radiation pattern using farfield fcn:
        if radiation_MT_phase=="P":
            disp = farfield(ned_mt, radiation_field_sample_pts, type="P") # Gets radiation displacement vector
            disp_magn = np.sum(disp * radiation_field_sample_pts, axis=0) # Magnitude of displacement (alligned with radius)  ???np.sqrt???
        elif radiation_MT_phase=="S":
            disp = farfield(ned_mt, radiation_field_sample_pts, type="S") # Gets radiation displacement vector
            disp_magn = np.sqrt(np.sum(disp * disp, axis=0)) # Magnitude of displacement (perpendicular to radius)
        disp_magn /= np.max(np.abs(disp_magn)) # Normalised magnitude of displacemnet

        # And convert radiation pattern to 2D coords:
        X_radiaton_coords,Y_radiaton_coords = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(radiation_field_sample_pts[0],radiation_field_sample_pts[1],radiation_field_sample_pts[2])

        # Create 2D XY radial mesh coords (for plotting) and plot radiation pattern:
        theta_spacing = theta[1]-theta[0]
        phi_spacing = phi[1]-phi[0]
        patches = []
        # Plot majority of radiation points as polygons:
        for b in range(len(disp_magn)):
            # Get coords at half spacing around point:
            theta_tmp = np.array([theta_flattened[b]-(theta_spacing/2.), theta_flattened[b]+(theta_spacing/2.)])
            phi_tmp = np.array([phi_flattened[b]-(phi_spacing/2.), phi_flattened[b]+(phi_spacing/2.)])
            # And check that doesn't go outside boundaries:
            if theta_flattened[b] == 0. or theta_flattened[b] == np.pi:
                continue # ignore as outside boundaries
            if phi_flattened[b] == 0.:# or phi_flattened[b] == 2*np.pi:
                continue # ignore as outside boundaries
            THETA_tmp, PHI_tmp = np.meshgrid(theta_tmp, phi_tmp)
            R_tmp = np.ones(4,dtype=float)
            x,y,z = convert_spherical_coords_to_cartesian_coords(R_tmp,THETA_tmp.flatten(),PHI_tmp.flatten())
            X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            # And plot (but ONLY if within bounding circle):
            if bounding_circle_path.contains_point((X[0],Y[0]), radius=0):
                poly_corner_coords = [(Y[0],X[0]), (Y[2],X[2]), (Y[3],X[3]), (Y[1],X[1])]
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.jet(int(disp_magn[b]*256)), alpha=0.6)
                ax.add_patch(polygon_curr)
        # Plot final point (theta,phi=0,0) (beginning point):
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.jet(int(disp_magn[0]*256)), alpha=0.6)
        ax.add_patch(centre_area)
        # Last point:
        #Circle([0.,0.], facecolor=matplotlib.cm.jet(int(disp_magn[0]*256)), alpha=0.6) # First point
    
    # Plot MT nodal plane solutions:
    # Get sample to plot:
    if len(MTs_to_plot[0,:]) > num_MT_solutions_to_plot:
        sample_indices = random.sample(range(len(MTs_to_plot[0,:])),num_MT_solutions_to_plot) # Get random sample of MT solutions to plot
    else:
        sample_indices = range(len(MTs_to_plot[0,:]))
    # Loop over MT solutions, plotting nodal planes:
    for i in sample_indices:
        # Get current mt:
        ned_mt = MTs_to_plot[:,i]
        
        # Get 3D nodal planes:
        plane_1_3D, plane_2_3D = get_nodal_plane_xyz_coords(ned_mt)
        # And switch vertical if neccessary:
        if lower_upper_hemi_switch=="upper":
            plane_1_3D[2,:] = -1*plane_1_3D[2,:] # as positive z is down, therefore down gives spherical projection
            plane_2_3D[2,:] = -1*plane_2_3D[2,:] # as positive z is down, therefore down gives spherical projection
        # And convert to 2D:
        X1,Y1 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(plane_1_3D[0],plane_1_3D[1],plane_1_3D[2])
        X2,Y2 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(plane_2_3D[0],plane_2_3D[1],plane_2_3D[2])
        
        # Get only data points within bounding circle:
        path_coords_plane_1 = [] # list to store path coords
        path_coords_plane_2 = [] # list to store path coords
        for j in range(len(X1)):
            path_coords_plane_1.append((X1[j],Y1[j]))
        for j in range(len(X2)):
            path_coords_plane_2.append((X2[j],Y2[j]))
        try:
            path_coords_plane_1_within_bounding_circle = np.vstack([p for p in path_coords_plane_1 if bounding_circle_path.contains_point(p, radius=0)])
            path_coords_plane_2_within_bounding_circle = np.vstack([p for p in path_coords_plane_2 if bounding_circle_path.contains_point(p, radius=0)])
            path_coords_plane_1_within_bounding_circle = np.vstack((path_coords_plane_1_within_bounding_circle, path_coords_plane_1_within_bounding_circle[0,:])) # To make no gaps
            path_coords_plane_2_within_bounding_circle = np.vstack((path_coords_plane_2_within_bounding_circle, path_coords_plane_2_within_bounding_circle[0,:])) # To make no gaps
            X1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,0]
            Y1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,1]
            X2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,0]
            Y2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,1]
        except ValueError:
            print "(Skipping nodal plane solution",i,"as can't plot.)"
            continue
        
        # And plot 2D nodal planes:
        alpha_nodal_planes = 0.3
        # Plot plane 1:
        for a in range(len(X1_within_bounding_circle)-1):
            if np.abs(Y1_within_bounding_circle[a]-Y1_within_bounding_circle[a+1])<0.25 and np.abs(X1_within_bounding_circle[a]-X1_within_bounding_circle[a+1])<0.25:
                ax.plot([Y1_within_bounding_circle[a], Y1_within_bounding_circle[a+1]],[X1_within_bounding_circle[a], X1_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
        # And plot plane 2:
        for a in range(len(X2_within_bounding_circle)-1):
            if np.abs(Y2_within_bounding_circle[a]-Y2_within_bounding_circle[a+1])<0.25 and np.abs(X2_within_bounding_circle[a]-X2_within_bounding_circle[a+1])<0.25:
                ax.plot([Y2_within_bounding_circle[a], Y2_within_bounding_circle[a+1]],[X2_within_bounding_circle[a], X2_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
        print "Plotted solution", i
    
    # Plot stations (if provided):
    if not len(stations) == 0:
        # Loop over stations:
        for station in stations:
            # Get params for station:
            azi=(station[1][0][0]/360.)*2.*np.pi + np.pi
            toa=(station[2][0][0]/360.)*2.*np.pi
            station_name = station[0][0]
            polarity = station[3][0][0]
            # And get 3D coordinates for station (and find on 2D projection):
            theta = np.pi - toa # as +ve Z = down
            phi = azi
            if theta>np.pi/2.:
                theta = theta - np.pi
                phi=phi+np.pi
            r = 1.0 # as on surface of focal sphere
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            # And plot based on polarity:
            if polarity == 1:
                ax.scatter(Y,X,c="r",marker="^",s=30,alpha=1.0)
            elif polarity == -1:
                ax.scatter(Y,X,c="b",marker="v",s=30,alpha=1.0)
            elif polarity == 0:
                ax.scatter(Y,X,c="w",marker='o',s=30,alpha=1.0)
            # And plot station name:
            plt.text(Y,X,station_name,color="k", fontsize=10, horizontalalignment="left", verticalalignment='top',alpha=1.0)
            
    # Finish plotting:
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_xlim(-2.0,2.0)
    ax.set_ylim(-2.0,2.0)
    plt.plot([-2.,2.],[0.,0.],c="k", alpha=0.5)
    plt.plot([0.,0.],[-2.,2.],c="k", alpha=0.5)
    plt.axis('off')
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()


# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Specify MT data dir (containing MTINV solutions):
    MT_data_filenames = glob.glob("MT_data/*.mat")
    
    # Loop over MT event files:
    for MT_data_filename in MT_data_filenames:
        
        print "Processing data for:", MT_data_filename
        
        # Import MT data:
        #MT_data_filename = "MT_data/20140629184210363MT.mat"
        uid, MTp, MTs, stations = load_MT_dict_from_file(MT_data_filename)

        # Find and plot sample of MTs data on Lune and return MT data from fitted gaussian:
        figure_filename = "Plots_out/"+MT_data_filename.split("/")[1].split(".")[0]+"_Lune.png"
        MTs_max_gau_loc = plot_sampled_MT_solns_on_Lune_with_gaussian_fit(MTs, MTp, frac_to_sample=0.1, figure_filename=figure_filename) #frac_to_sample=0.1

        # Get most likely solution:
        index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
        MT_max_prob = MTs[:,index_MT_max_prob]
        # And get full MT matrix:
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
        print "Full MT (max prob.):"
        print full_MT_max_prob
        print "(For plotting radiation pattern)"

        # Plot MT solutions and radiation pattern of most likely on sphere:
        #MTs_to_plot = np.reshape(MT_max_prob, (6,1))
        MTs_to_plot = MTs_max_gau_loc#[:,0:15]
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        figure_filename = "Plots_out/"+MT_data_filename.split("/")[1].split(".")[0]+".png"
        plot_MTs_on_sphere_twoD(MTs_to_plot, radiation_pattern_MT=radiation_pattern_MT, stations=stations, radiation_MT_phase = "P", lower_upper_hemi_switch="lower", figure_filename=figure_filename, num_MT_solutions_to_plot=20)

        print "Finished processing data for:", MT_data_filename
    
    print "Finished"
        
