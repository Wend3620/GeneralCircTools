import xarray as xr, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units

def EPflux(dataset, divergence = True, QG = False, rho_fix = False):
    r'''
    Parameters
    ----------
    dataset: 'xarray.Dataset' 
        The dataset for EP flux calculation.
        The dataset should contain following parameters: u(zonal wind), v(meridional wind), t(temperature).
    divergence: 'bool'
        Whether to calculate EP flux divergence or not. Default is True

    QG: 'bool'
        Use QG version equation for calculation if True, not if False

    rho_fix: 'bool'
        If True, the function will use rho(density) as one of the scaling factor for the Flux vector.
    
    Returns
    -------
    'xarray.Dataset'
        A dataset contains calculated data. 
    '''
    thta = False
    for i in list(dataset.coords):
        if i.lower() in 'latitudesxzonal' and i!='lat':
            dataset=dataset.rename({i: "lat"})
        elif i.lower() in 'longitudesymeridional' and i != 'lon':
            dataset=dataset.rename({i: "lon"})
        elif i.lower() in 'pressureisobaricinhpa' and  i != 'pressure':
            dataset=dataset.rename({i: "pressure"})
        elif i.lower() in 'temperature' and i != 't':
            dataset=dataset.rename({i: "t"})
        elif i.lower() in ['potential temperature', 'theta', 'thta']:
            thta = True
            if i != 'thta':
                dataset=dataset.rename({i: "thta"})
        else: 
            continue
    if thta == False:
        dataset['thta']=mpcalc.potential_temperature(dataset.pressure, dataset.t).transpose('time', 'pressure', 'lat', 'lon')
    ds_bar = dataset.mean('lon')

    T0= dataset.t.sel(pressure = 1000)#K
    R0 = 287 * units('J/kg/kelvin') #J/kg/K
    P0= 101300 *units('Pa')#hPa
    a = 6378000 * units.meter #m radius of the earth
    rho0 = P0/T0/R0 #kg/m3
    g = 9.81 * units('m/s^2')
    omega = 2*np.pi/(24*3600)
    f = 2*omega*np.sin(dataset.lat/180*np.pi) * units("s^-1")
    cphi = np.cos(dataset.lat*np.pi/180)
    sphi = np.sin(dataset.lat*np.pi/180)
    rho = rho0*(dataset.pressure*100*units.Pa)/P0
    rho.attrs["long_name"] = "density"
    rho.attrs["units"] = "kg/m3"
    rho.attrs["standard_name"] = "air_density"
    rho = rho*units("joule/Pa/m^3")

    u_v = (dataset.u*dataset.v).mean('lon')
    up_vp = u_v - ds_bar.u*ds_bar.v

    thta_v = (thta*dataset.v).mean('lon')
    thtap_vp = thta_v - ds_bar.thta*ds_bar.v

    dp = np.gradient(ds_bar.pressure, edge_order=2)
    dp = xr.DataArray(dp, dims = ['pressure'], coords=dict(pressure = dataset.pressure), name='pressure')*-100
    dz = dp*R0*dataset.t/g/dataset.pressure/100

    dthta = np.gradient(ds_bar.thta, axis=1, edge_order=2)
    dthta = xr.DataArray(dthta*-1, dims = ['time', 'pressure', 'lat'],
                        coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))
    dthtadp = dthta/dp * units('kelvin/Pa')

    dphi = np.gradient(dataset.lat, edge_order=2)
    dphi = dphi*np.pi/-180
    dphi = xr.DataArray(dphi, dims = ['lat'], coords=dict(lat = dataset.lat), name='dphi')

    if QG != True:
        dudp = np.gradient(dataset.u, axis = 1, edge_order=2)
        dudp = xr.DataArray(dudp*-1, dims = ['time', 'pressure','lat', 'lon'],
                            coords=dict(pressure = dataset.pressure, lat= dataset.lat, time = dataset.time))*units('m/s/Pa')
        dudp = (dudp/dp).mean('lon')
        Ep1ag = dudp*thtap_vp/dthtadp
        Ep1 = -1*up_vp + Ep1ag
    else:
        Ep1 = -1*up_vp

    if rho_fix != False:
        Fy = rho*a*cphi*Ep1
    else:
        Fy = a*cphi*Ep1

    if QG != True:
        du = np.gradient(ds_bar.u, axis = 2, edge_order = 2)
        #print(du.shape)
        du = xr.DataArray(du*-1, dims = ['time', 'pressure', 'lat'],
                            coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))
        subf = ((du.mean('time')/dphi-sphi)/a/cphi)*units('m/s') #Fp term2
        up_wp = ((dataset.u - ds_bar.u)*(dataset.w-ds_bar.w)).mean('lon')
        Ep2ag = -1*subf*thtap_vp/dthtadp-up_wp
        Ep2 = thtap_vp/dthtadp*f + Ep2ag
    else:
        Ep2 = thtap_vp/dthtadp*f

    Ep2 = Ep2*-1
    if rho_fix != False:
        Fz = rho*a*cphi*Ep2/dp*dz
    else:
        Fz = a*cphi*Ep2/dp*dz
    
    Fy1.name = 'Fy'
    Fz1.name = 'Fz'

    if divergence == True:
        dEp1 = np.gradient(Ep1 , axis = 2, edge_order=2)
        dEp1 = xr.DataArray(-1*dEp1, dims = ['time', 'pressure','lat'],
                            coords=dict(pressure = dataset.pressure, lat= dataset.lat, time = dataset.time))
        dEp1 = dEp1/dphi*cphi - 2*sphi*Ep1*units('s^2/m^2')
        dFy2 = dEp1/a/cphi
        dFy2 = dFy2*units('m^2/s/day')

        dEp2 = np.gradient(Ep2, axis = 1, edge_order=2)
        dEp2 = xr.DataArray(dEp2*-1, dims = ['time', 'pressure','lat'],
                            coords=dict(pressure = dataset.pressure, lat= dataset.lat, time = dataset.time))
        dFz2 = dEp2/dp
        dFz2 = dFz2*units('m/s/day') #* units('joule/kg*s/day')

        delF = (dFy2+dFz2)*86400
        
        Fy1 = Fy.mean(['time', 'lon']).transpose()
        Fz1 = (Fz.mean(['time', 'lon'])*units('kg*m^2/s^2/joule/pascal')).transpose()
        if rho_fix != False:
            delF1 = (delF*rho).mean(['time','lon'])
        else:
            delF1 = delF.mean('time')
        delF1.name = 'delF'
        out = xr.merge([Fy1, Fz1, delF1])
    else: 
        out = xr.merge([Fy1, Fz1])
    return out