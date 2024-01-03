import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import xarray as xr
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import os
from scipy.interpolate import LinearNDInterpolator


def find_zero(vec):
    "This function finds the index in a vector where it is the first time that the value change the sign"
    for n in range(len(vec)):
        if vec[n]*vec[n+1]<=0:
            break
    return n 

def find_closest(latNC, lonNC,latP,lonP):
    "This function will give you the indexes in the lat and long vectors, between the latP and lonP is"
    diffLat = latNC-latP
    diffLon = lonNC- lonP
    idx_lat = find_zero(diffLat)
    idx_lon = find_zero(diffLon)
    return idx_lat,idx_lat+1,idx_lon,idx_lon+1

def interpolate_4points(x,y,z,Xpto,Ypto):
    "This calculate linear interpolation using x,y points with z values in the Xpto,Ypto"
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z = interp(Xpto,Ypto)
    return Z

    
def spatial_interpolation (ds,variable,lat_point,lon_point,level,lon_name='longitude',lat_name='latitude'):
    "This function returns the interpolated (linear interpolation) values of the variable(enter the string name in the ds xarray) from the xrray ds, in the point determined by the lat_point, lon_point"
    # Get the index in lat, and long closest to the lat_point,lon_point
    lat1,lat2,lon1,lon2 = find_closest(ds.coords[lat_name].data,ds.coords[lon_name].data,lat_point,lon_point) # these are the index in lat, and lon between where the point is

    # these are the lats and long values that will be interpolated
    lats_selected = np.array([ds.coords[lat_name].data[lat1],
                            ds.coords[lat_name].data[lat1],
                            ds.coords[lat_name].data[lat2],
                            ds.coords[lat_name].data[lat2]])

    lons_selected = np.array([ds.coords[lon_name].data[lon1],
                            ds.coords[lon_name].data[lon2],
                            ds.coords[lon_name].data[lon1],
                            ds.coords[lon_name].data[lon2]])
    if level != None:
    # I get the data in those point for the variable
        data_df = pd.DataFrame({'Time':ds.coords['time'].data,
                                'Point 1':ds[variable].sel(latitude=ds.coords[lat_name].data[lat1],longitude = ds.coords[lon_name].data[lon1],level=level).data,
                                'Point 2': ds[variable].sel(latitude=ds.coords[lat_name].data[lat1],longitude = ds.coords[lon_name].data[lon2],level=level).data,
                                'Point 3': ds[variable].sel(latitude=ds.coords[lat_name].data[lat2],longitude = ds.coords[lon_name].data[lon1],level=level).data,
                                'Point 4': ds[variable].sel(latitude=ds.coords[lat_name].data[lat2],longitude= ds.coords[lon_name].data[lon2],level=level).data})
    else:
         data_df = pd.DataFrame({'Time':ds.coords['time'].data,
                                'Point 1':ds[variable].sel(latitude=ds.coords[lat_name].data[lat1],longitude = ds.coords[lon_name].data[lon1]).data,
                                'Point 2': ds[variable].sel(latitude=ds.coords[lat_name].data[lat1],longitude = ds.coords[lon_name].data[lon2]).data,
                                'Point 3': ds[variable].sel(latitude=ds.coords[lat_name].data[lat2],longitude = ds.coords[lon_name].data[lon1]).data,
                                'Point 4': ds[variable].sel(latitude=ds.coords[lat_name].data[lat2],longitude= ds.coords[lon_name].data[lon2]).data})
    # Now I interpolate spatially 
    interpolate = data_df.apply(lambda row: interpolate_4points(lats_selected,lons_selected,row[1:5].values,lat_point,lon_point),axis=1).astype('float')
    return interpolate

def atm_profile (file,variable,date,levels):
    """"This function returns the profile in the date, and it takes in the defines levels for the variable, taking the data from the dictionary levels, where the keys are the levels in pressure"""
    profile = []
    for level in levels:
        data = file[level]
        profile.append(data[data['Time']==date][variable].values[0])
    return profile