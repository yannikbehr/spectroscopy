#Copyright (C) Nial Peters 2015
#
#This file is part of gns_flyspec.
#
#gns_flyspec is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#gns_flyspec is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.
from collections import namedtuple
import numpy
import scipy.optimize
import math
import warnings



GaussianParameters = namedtuple('GaussianParameters',['amplitude','mean',
                                                      'sigma','y_offset'])

def gaussian_pts(xpts, p):
    """
    Returns the values of a Gaussian (described by the GaussianParameters p)
    corresponding to the x values passed in as xpts (an array).
    """
    xpts = numpy.array(xpts)
    f = gaussian_func(p)
    
    return f(xpts)

def gaussian_func(p):
    return lambda x: p[0]*scipy.exp(-(x-p[1])**2/(2.0*p[2]**2)) + p[3]


class FittingError(RuntimeError):
    pass


def __errfunc(p,x,y):
    #define an error function such that there is a high cost to having a 
    #negative amplitude
    diff = (p[0]*scipy.exp(-(x-p[1])**2/(2.0*p[2]**2)) + p[3]) - y
    if p[0] < 0.0:
        diff *= 10000.0

    return diff

#function taken from avoscan.processing module
def fit_gaussian(xdata, ydata, amplitude_guess=None, mean_guess=None, 
                 sigma_guess=None, y_offset_guess=None):
    """
    Fits a gaussian to some data using a least squares fit method. Returns a named tuple
    of best fit parameters (amplitude, mean, sigma, y_offset).
    
    Initial guess values for the fit parameters can be specified as kwargs. Otherwise they
    are estimated from the data.
    
    If plot_fit=True then the fit curve is plotted over the top of the raw data and displayed.
    """
    
    if len(xdata) != len(ydata):
        raise ValueError, "Lengths of xdata and ydata must match"
    
    if len(xdata) < 4:
        raise ValueError, "xdata and ydata need to contain at least 4 elements each"
    
    # guess some fit parameters - unless they were specified as kwargs
    if amplitude_guess is None:
        amplitude_guess = max(ydata)
    
    if mean_guess is None:
        weights = ydata - numpy.average(ydata)
        weights[numpy.where(weights <0)]=0 
        mean_guess = numpy.average(xdata,weights=weights)
                   
    #use the y value furthest from the maximum as a guess of y offset 
    if y_offset_guess is None:
        data_midpoint = (xdata[-1] + xdata[0])/2.0
        if mean_guess > data_midpoint:
            yoffset_guess = ydata[0]        
        else:
            yoffset_guess = ydata[-1]

    #find width at half height as estimate of sigma        
    if sigma_guess is None:      
        variance = numpy.dot(numpy.abs(ydata), (xdata-mean_guess)**2)/numpy.abs(ydata).sum()  # Fast and numerically precise    
        sigma_guess = math.sqrt(variance)
    
    
    #put guess params into an array ready for fitting
    p0 = numpy.array([amplitude_guess, mean_guess, sigma_guess, yoffset_guess])

    # do the fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1, success = scipy.optimize.leastsq(__errfunc, p0, args=(xdata,ydata))
        
    if success not in (1,2,3,4):
        raise FittingError("Could not fit Gaussian to data.")
    
    return GaussianParameters(*p1)


