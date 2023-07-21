import xarray as xr, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units

def EPflux(dataset, divergence = True, QG = False):

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

    T0= dataset.t.sel(pressure = 1000)#K
    R0 = 287 * units('J/kg/kelvin') #J/kg/K
    P0= 100000 *units('Pa')#hPa
    a = 6378000 * units.meter #m radius of the earth
    rho0 = P0/T0/R0 #kg/m3
    g = 9.81 * units('m/s^2')
    omega = 2*np.pi/(24*3600)
    f = 2*omega*np.sin(dataset.lat/180*np.pi) * units("s^-1")
    cphi = np.cos(dataset.lat*np.pi/180)
    sphi = np.sin(dataset.lat*np.pi/180)
    rho = rho0*(dataset.pressure*100*units.Pa)/P0
    rho = rho*units("joule/Pa/m^3")
    ds_bar = dataset.mean('lon')

    #Calculate eddy
    up_vp = (dataset.u*dataset.v).mean('lon') - ds_bar.u*ds_bar.v
    thtap_vp = (dataset.thta*dataset.v).mean('lon') - ds_bar.thta*ds_bar.v

    dpdz = -1*(dataset.pressure*100)*units('Pa')/dataset.t*g/R0*units('(joule second^2)/(kilogram*meter^2)') #Rearrange units

    dthtadp = ds_bar.thta.differentiate(coord='pressure', edge_order=2)*units('1/Pa')/100
    up_wp = ((dataset.u - ds_bar.u)*(dataset.w-ds_bar.w)).mean('lon')
    if QG != True:
        dudp = ds_bar.u.differentiate(coord='pressure', edge_order=2)*units('1/Pa')/100
        Ep1ag = dudp*dpdz*thtap_vp/(dthtadp*dpdz)
        fy = -1*up_vp + Ep1ag
    else:
        fy = -1*up_vp
    Fy = (rho*a*cphi*fy)

    if QG != True:
        dudphi = ds_bar.u.differentiate(coord='lat', edge_order=2)*180/np.pi
        subf = (dudphi/a-sphi/a/cphi*ds_bar.u) #Fp term2
        Ep2ag = -1*subf*thtap_vp/dthtadp-up_wp
        fp = thtap_vp/dthtadp*f + Ep2ag
    else: 
        fp = thtap_vp/dthtadp*f
    fz = fp/dpdz
    Fz = rho*a*cphi*fz

    Fy.name = 'Fy'
    Fz.name = 'Fz'

    if divergence == True:
        dFy1dphi = Fy.differentiate(coord='lat', edge_order=2)*180/np.pi
        dEp1 = dFy1dphi/a - 2*rho*sphi*fy
        dFy = dEp1/a/rho/cphi*units('s/day')*86400
        dFz = Fz.differentiate(coord='pressure', edge_order=2)*units('1/Pa')/100*dpdz/a/rho/cphi*units('s/day')*86400

        dFy.name = 'dFy'
        dFz.name = 'dFz'
         
        return xr.merge([Fy, Fz, dFz, dFy])
    else:
        return xr.merge([Fy, Fz])