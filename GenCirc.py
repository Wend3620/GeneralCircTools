import xarray as xr, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units

def EPflux(dataset, divergence = True, QG = False):
    
    r'''
    (Using Xarray derivatives)
    Function used for calculating Eliassen Palm flux (EP flux) vectors.
    
    Variable required: Zonal wind(u), Meridional wind(v), Temperature(t).
    
    Dimension required: Pressure (p), Latitude(lat), Longitude(lon).
    
    Parameters
    ----------
    dataset: 'xarray.Dataset' 
        The dataset for EP flux calculation.
        The dataset should contain following parameters: u(zonal wind), v(meridional wind), t(temperature).
    divergence: 'bool'
        Whether to calculate EP flux divergence or not. Default is True

    QG: 'bool'
        Use QG version equation for calculation if True, not if False
    
    Returns
    -------
    'xarray.Dataset'
        A dataset contains calculated data in 3 dimensions (Latitude, Pressure, Time). 
    '''
    thta = False
    dim_ready = 0
    for i in list(dataset.coords):
        test_string = i.lower().replace('_', '').replace(' ', '')
        if test_string in 'latitudesxzonal': 
            dim_ready+=1
            if i!='lat':
                dataset=dataset.rename({i: "lat"})
        elif test_string in 'longitudesymeridional':
            dim_ready+=1
            if i != 'lon':
                dataset=dataset.rename({i: "lon"})
        elif test_string in 'pressureisobaricinhpa': 
            dim_ready+=1
            if i != 'pressure':
                dataset=dataset.rename({i: "pressure"})
        elif test_string in 'uwindzonalwind':
            dim_ready+=1
            if i != 'u':
                dataset=dataset.rename({i: "u"})
        elif test_string in 'vwindmeridionalwind' :
            dim_ready+=1
            if i != 'v':
                dataset=dataset.rename({i: "v"})
       
        elif test_string in 'potentialtemperaturethetathta':
            dim_ready+=1
            thta = True
            if test_string in 'temperature' :
                thta = False 
                if i != 't':
                    dataset=dataset.rename({i: "t"})    
            elif i != 'thta':
                dataset=dataset.rename({i: "thta"})
            
        else: 
            continue

    if thta == False:
        dataset['thta']=mpcalc.potential_temperature(dataset.pressure, dataset.t).transpose('time', 'pressure', 'lat', 'lon')

    if dim_ready != 6:
        raise KeyError("You are missing one of the dimensions required for the calculation")

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
        print(str((dataset['time'].values)[1]) + ': done')
        return xr.merge([Fy, Fz, dFz, dFy]).mean(['lon'])
    else:
        print(str((dataset['time'].values)[1]) + ': done')
        return xr.merge([Fy, Fz]).mean(['lon'])
    

def EPflux_np(dataset, divergence = True, QG = False):

    r'''
    (Using numpy to calculate derivatives)
    Function used for calculating Eliassen Palm flux (EP flux) vectors.
    
    Variable required: Zonal wind(u), Meridional wind(v), Temperature(t), Vertical wind(w)(Ignored if you want QG results only(QG==True)).
    
    Dimension required: Pressure (p), Latitude(lat), Longitude(lon).
    
    Parameters
    ----------
    dataset: 'xarray.Dataset' 
        The dataset for EP flux calculation.
    divergence: 'bool'
        Whether to calculate EP flux divergence or not. Default is True
    QG: 'bool'
        Use QG version equation for calculation if True, not if False
    
    Returns
    -------
    'xarray.Dataset'
        A dataset contains calculated data in 3 dimensions (Latitude, Pressure, Time).
    '''
    thta = False
    dim_ready = 0
    for i in list(dataset.coords):
        test_string = i.lower().replace('_', '').replace(' ', '')
        if test_string in 'latitudesxzonal': 
            dim_ready+=1
            if i!='lat':
                dataset=dataset.rename({i: "lat"})
        elif test_string in 'longitudesymeridional':
            dim_ready+=1
            if i != 'lon':
                dataset=dataset.rename({i: "lon"})
        elif test_string in 'pressureisobaricinhpa': 
            dim_ready+=1
            if i != 'pressure':
                dataset=dataset.rename({i: "pressure"})
        elif test_string in 'uwindzonalwind':
            dim_ready+=1
            if i != 'u':
                dataset=dataset.rename({i: "u"})
        elif test_string in 'vwindmeridionalwind' :
            dim_ready+=1
            if i != 'v':
                dataset=dataset.rename({i: "v"})
        elif test_string in 'potentialtemperaturethetathta':
            dim_ready+=1
            thta = True
            if test_string in 'temperature' :
                thta = False 
                if i != 't':
                    dataset=dataset.rename({i: "t"})    
            elif i != 'thta':
                dataset=dataset.rename({i: "thta"})
        elif test_string in 'wwindverticalwind' and QG == True:
            dim_ready+=1
            if i != 'w':
                dataset=dataset.rename({i: "w"})
        else: 
            continue

    if thta == False:
        dataset['thta']=mpcalc.potential_temperature(dataset.pressure, dataset.t).transpose('time', 'pressure', 'lat', 'lon')
        
    if QG == True and dim_ready != 7 :
            raise KeyError("You are missing one of the dimensions required for the calculation")
    elif QG == False and dim_ready != 6 :
            raise KeyError("You are missing one of the dimensions required for the calculation")
        
    #Constants
    T0= dataset.t.sel(pressure = 1000)#K
    R0 = 287 * units('m^2/s^2/kelvin') #J/kg/K
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
    rho = rho*units("kg/Pa/s^2/m")
    #Zonal mean
    ds_bar = dataset.mean('lon')
    #Eddies
    u_v = (dataset.u*dataset.v).mean('lon')
    up_vp = u_v - ds_bar.u*ds_bar.v
    thta_v = (dataset.thta*dataset.v).mean('lon')
    thtap_vp = thta_v - ds_bar.thta*ds_bar.v
    #Derivatives
    dp = np.gradient(ds_bar.pressure, edge_order=2)
    dp = (xr.DataArray(dp, dims = ['pressure'], coords=dict(pressure = dataset.pressure), name='pressure')*-100)*units('Pa')

    dz = (dp*R0*dataset.t/g/dataset.pressure/100)*units('1/Pa')
    dthta = np.gradient(ds_bar.thta, axis=1, edge_order=2)

    dthta = xr.DataArray(dthta*-1, dims = ['time', 'pressure', 'lat'],
                        coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))

    dthtadp = dthta/dp * units('kelvin')
    dphi = np.gradient(dataset.lat, edge_order=2)

    dphi = dphi*np.pi/-180
    dphi = xr.DataArray(dphi, dims = ['lat'], coords=dict(lat = dataset.lat), name='dphi')
    #Putting everything together...
    if QG != True:
        dudp = np.gradient(dataset.u, axis = 1, edge_order=2)
        #print(dudp.shape)
        dudp = xr.DataArray(dudp*-1, dims = ['time', 'pressure','lat', 'lon'],
                            coords=dict(pressure = dataset.pressure, lat= dataset.lat, time = dataset.time))*units('m/s')
        dudp = (dudp/dp).mean('lon')
        Ep1ag = dudp*thtap_vp/dthtadp
        fy = -1*up_vp + Ep1ag
    else:
        fy = -1*up_vp

    Fy = rho*a*cphi*fy
    

    if QG != True:
        du = np.gradient(ds_bar.u, axis = 2, edge_order = 2)
        du = xr.DataArray(du*-1, dims = ['time', 'pressure', 'lat'], 
                        coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))
        subf = ((du.mean('time')*cphi/dphi*units('m/s')-ds_bar.u*sphi)/a/cphi) #Fp term2
        up_wp = ((dataset.u - ds_bar.u)*(dataset.w-ds_bar.w)).mean('lon')
        Ep2ag = -1*subf*thtap_vp/dthtadp-up_wp
        fp = thtap_vp/dthtadp*f + Ep2ag
    else:
        fp = thtap_vp/dthtadp*f

    fz = fp/dp*dz*-1
    Fz = rho*a*cphi*fz

    Fy = Fy.mean('lon')
    Fz = Fz.mean(['lon'])

    Fy.name = 'Fy'
    Fz.name = 'Fz'
    
    if divergence == True:
        dfy = np.gradient(fy * rho.mean('lon') , axis = 2, edge_order=2)
        dfy = xr.DataArray(-1*dfy, dims = ['time', 'pressure','lat'],
                            coords=dict(pressure = dataset.pressure, lat=dataset.lat, time = dataset.time))
        dfy = dfy/dphi*cphi - 2*sphi*fy * rho * units ('s^2*m/kg') #* units('s^2/m^2')
        dFy = dfy/rho/a/cphi*units('kg/m^3')
        dFy = 86400*dFy.mean('lon')*units('m^2/s/day')

        dfz = np.gradient((fz * rho).mean('lon'), axis = 1, edge_order=2)
        dfz = xr.DataArray(dfz, dims = ['time', 'pressure','lat'], 
                        coords=dict(pressure = dataset.pressure, lat=dataset.lat, time = dataset.time))
        dFz = dfz/rho/dz*units('kg/m^3')
        dFz = 86400*dFz.mean('lon')*units('m^2/s/day') #* units('joule/kg*s/day')

        dFy.name = 'dFy'
        dFz.name = 'dFz'
        print(str((dataset['time'].values)[1]) + ': done')
        return xr.merge([Fy, Fz, dFy, dFz])
    else: 
        print(str((dataset['time'].values)[1]) + ': done')
        return xr.merge([Fy, Fz])

# def EVector(dataset, divergence = True, QG = False):

#     r'''
#     (Using numpy to calculate derivatives)
#     Function used for calculating Eliassen Palm flux (EP flux) vectors.
    
#     Variable required: Zonal wind(u), Meridional wind(v), Temperature(t), Vertical wind(w)(Ignored if you want QG results only(QG==True)).
    
#     Dimension required: Pressure (p), Latitude(lat), Longitude(lon).
    
#     Parameters
#     ----------
#     dataset: 'xarray.Dataset' 
#         The dataset for EP flux calculation.
#     divergence: 'bool'
#         Whether to calculate EP flux divergence or not. Default is True
#     QG: 'bool'
#         Use QG version equation for calculation if True, not if False
    
#     Returns
#     -------
#     'xarray.Dataset'
#         A dataset contains calculated data in 3 dimensions (Latitude, Pressure, Time).
#     '''
    
#     QG = False
#     T0= dataset.t.sel(pressure = 1000)#K
#     R0 = 287 * units('m^2/s^2/kelvin') #J/kg/K
#     P0= 101300 *units('Pa')#hPa
#     # a = 6378000 * units.meter #m radius of the earth
#     rho0 = P0/T0/R0 #kg/m3
#     cphi = np.cos(dataset.lat*np.pi/180)
#     rho = rho0*(dataset.pressure*100*units.Pa)/P0
#     rho.attrs["long_name"] = "density"
#     rho.attrs["units"] = "kg/m3"
#     rho.attrs["standard_name"] = "air_density"
#     rho = rho*units("kg/Pa/s^2/m")
#     #Zonal mean
#     ds_bar = dataset.mean('lon')
#     #Eddies
#     up = dataset.u - ds_bar.u
#     vp = dataset.v - ds_bar.v
#     up_vp = up*vp
    
#     thtap_vp = (dataset.v - ds_bar.v)*(dataset.thta - ds_bar.thta)
#     #Derivatives
#     dp = np.gradient(ds_bar.pressure, edge_order=2)
#     dp = (xr.DataArray(dp, dims = ['pressure'], coords=dict(pressure = dataset.pressure), name='pressure')*-100)*units('Pa')

#     dphi = np.gradient(dataset.lat, edge_order=2)

#     dphi = dphi*np.pi/-180
#     dphi = xr.DataArray(dphi, dims = ['lat'], coords=dict(lat = dataset.lat), name='dphi')
    
#     dthta = xr.DataArray(dthta*-1, dims = ['time', 'pressure', 'lat'],
#                     coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))

#     dthtadp = dthta/dp * units('kelvin')
#     #Putting everything together...
#     fx = 0.5*((vp**2).mean('time') - (up**2).mean('time'))
#     Fx = rho*cphi*fx #*a
    
#     if QG != True:
#         dudp = np.gradient(dataset.u, axis = 1, edge_order=2)
#         #print(dudp.shape)
#         dudp = xr.DataArray(dudp*-1, dims = ['time', 'pressure','lat', 'lon'],
#                             coords=dict(pressure = dataset.pressure, lat= dataset.lat, time = dataset.time))*units('m/s')
#         dudp = (dudp/dp)
#         Ep1ag = dudp*thtap_vp/dthtadp
#         fy = -1*up_vp + Ep1ag
#     else:
#         fy = -1*up_vp

#     Fy = rho*cphi*fy.mean('time') #*a

#     Fx.name = 'Fx'
#     Fy.name = 'Fy'
    
#     if divergence == True:
#         dfy = np.gradient(fy * rho.mean('lon') , axis = 2, edge_order=2)
#         dfy = xr.DataArray(-1*dfy, dims = ['time', 'pressure','lat'],
#                             coords=dict(pressure = dataset.pressure, lat=dataset.lat, time = dataset.time))
#         dfy = dfy/dphi*cphi - 2*sphi*cphi*fy * rho * units ('s^2*m/kg') #* units('s^2/m^2')
#         dFy = dfy/rho/a/cphi*units('kg/m^3')
#         dFy = 86400*dFy.mean('lon')*units('m^2/s/day')

#         dFy.name = 'dFy'
        
#         print(str((dataset['time'].values)[1]) + ': done')
#         return xr.merge([Fx, Fy, dFx, dFy])
#     else: 
#         print(str((dataset['time'].values)[1]) + ': done')
#         return xr.merge([Fx, Fy])

def TEM_vectors(dataset, pressure_level = False):
    r'''
    (Using numpy to calculate derivatives)
    Function used for calculating Transformed Eulerian Means(TEM) Circulation vectors.
    
    Variable required: Meridional wind(v), Vertical wind(w), Temperature(t) .
    
    Dimension required: Pressure (p), Latitude(lat), Longitude(lon).
    
    Parameters
    ----------
    dataset: 'xarray.Dataset' 
        The dataset for EP flux calculation.
    pressure_level: 'bool'
        Whether to have your vertical component results with unit m/s or Pa/s. Default is False (m/s)
    
    Returns
    -------
    'xarray.Dataset'
        A dataset contains calculated data (v_star, w_star) in 3 dimensions (Latitude, Pressure, Time).
    '''
    thta = False
    dim_ready = 0
    for i in list(dataset.coords):
        test_string = i.lower().replace('_', '').replace(' ', '')
        if test_string in 'latitudesxzonal': 
            dim_ready+=1
            if i!='lat':
                dataset=dataset.rename({i: "lat"})
        elif test_string in 'longitudesymeridional':
            dim_ready+=1
            if i != 'lon':
                dataset=dataset.rename({i: "lon"})
        elif test_string in 'pressureisobaricinhpa': 
            dim_ready+=1
            if i != 'pressure':
                dataset=dataset.rename({i: "pressure"})
        elif test_string in 'vwindmeridionalwind' :
            dim_ready+=1
            if i != 'v':
                dataset=dataset.rename({i: "v"})
       
        elif test_string in 'potentialtemperaturethetathta':
            dim_ready+=1
            thta = True
            if test_string in 'temperature' :
                thta = False 
                if i != 't':
                    dataset=dataset.rename({i: "t"})    
            elif i != 'thta':
                dataset=dataset.rename({i: "thta"})
            
        else: 
            continue

    if thta == False:
        dataset['thta']=mpcalc.potential_temperature(dataset.pressure, dataset.t).transpose('time', 'pressure', 'lat', 'lon')

    if dim_ready != 5:
        raise KeyError("You are missing one of the dimensions required for the calculation")
    
    #Calculation for mean values
    ds_bar = dataset.mean('lon')
    
    a = 6378000 * units.meter #m radius of the earth
    cphi = np.cos(dataset.lat*np.pi/180)
    H = 7000*units('m')

    #Eddies
    thta_v = (dataset.thta*dataset.v).mean('lon')
    thtap_vp = thta_v - ds_bar.thta*ds_bar.v
    #Derivatives
    dp = np.gradient(ds_bar.pressure, edge_order=2)
    dp = (xr.DataArray(dp, dims = ['pressure'], coords=dict(pressure = dataset.pressure), name='pressure')*-100)*units('Pa')

    dthta = np.gradient(ds_bar.thta, axis=1, edge_order=2)

    dthta = xr.DataArray(dthta*-1, dims = ['time', 'pressure', 'lat'],
                        coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))
    
    dthtadp = dthta/dp * units('kelvin')
    dphi = np.gradient(dataset.lat, edge_order=2)
    dphi = dphi*np.pi/-180
    dphi = xr.DataArray(dphi, dims = ['lat'], coords=dict(lat = dataset.lat), name='dphi')

    #Vertical component
    w_upper = np.gradient(thtap_vp/dthtadp*cphi, axis= 2, edge_order=2)*np.pi/-180
    w_upper= xr.DataArray(w_upper, dims = ['time', 'pressure', 'lat'],
            coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))*units('Pa*meter/second')
    w_TEM = ds_bar.w + w_upper/a/dphi/cphi
    if pressure_level == False:
            w_anom = w_anom*-1*H/(w_anom.pressure*100*units('Pa'))

    #Meridional component
    v_upper = np.gradient(thtap_vp/dthtadp, axis= 2, edge_order=2)
    v_upper= xr.DataArray(v_upper, dims = ['time', 'pressure', 'lat'],
            coords=dict(pressure = dataset.pressure, time = dataset.time, lat= dataset.lat))*units('Pa*meter/second')
    v_TEM = ds_bar.v + v_upper/dp
    
    w_TEM.name = 'w_star'
    v_TEM.name = 'v_star'
    return xr.merge([v_TEM, w_TEM])


        
def EOF(data, normalize=True):
    '''
    Function for performing EOF analysis
    Parameters: 
    data: array-like
        Your input data
    normalize: Boolean
        True if input data is expected to be normalized before the analysis.
    Returns:
    pc: array-like
        Resulted principal components
    angles: array-like
        Phase angle calculated by arctan(pc2/pc1) (Second component/First component)
    '''
    if normalize==True:
        raw_data = data #.sel(isobaricInhPa=50)
        X_mean = raw_data.mean()
        X_std = raw_data.std()
        data = (raw_data - X_mean) / X_std
    ucov = np.cov(data.T)
    eval, evec = np.linalg.eig(ucov)
    idxes = np.argsort(eval)[::-1]
    eval = eval[idxes]
    evec = evec[:, idxes]

    pc = np.dot(data, evec) #Principal component
    
    thta = np.arctan2(pc[:, 1], pc[:, 0])
    vals = np.degrees(thta)
    angles = np.array([float(i) if i>0 else float(i)+360 for i in vals])
    return pc, angles
