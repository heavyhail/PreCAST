#!/usr/bin/python3

# Code to calculate AR-CHaMo predictors from IFS model data
# Developed in the project PreCAST by Pieter Groenemeijer, Francesco Battaglioli and Ivan Tsonevsky (ECMWF)

import numpy as np
import metview as mv
import xarray as xr
import sys
from scipy.interpolate import RegularGridInterpolator as RGI

import tracemalloc

parcel_lookupfile = '/home/moit/ecFlow/archamo/lu_table_parcel.nc'

tracemalloc.start()

basetime = sys.argv[1]
step = sys.argv[2]
nmem = sys.argv[3]
datapath = sys.argv[4]

nstep = int(step)
if nstep <= 144:
    acc = 3
else:
    acc = 6
prev_nstep = nstep - acc
all_timesteps = [prev_nstep, nstep]

# Constants
ms2knots = 1.944
g = 9.81
rd = 287.04
rv = 461.50
eps = rd / rv
cpd = 1005.7

def main():

    fieldnames_3d = ['t', 'q', 'r', 'u', 'v', 'z']
    fieldnames_2d = ['10u', '10v', '200u', '200v', '2d', '2t', 'cp', 'sp', 'deg0l', 'mlcape50', 'mudlp', 'tp', 'z', 'lsm', 'mucape']
    fieldnames_2d_add = ['z', 'cp', 'tp']

    params_dict = {
        "MU_LI" : "bli",
        "RHmean": "r",
        "mcpr": "cprate",
        "lsm": "lsm",
        "MU_MIXR": "q",
        "MU_EFF_BS": "ws",
        "MU_CAPE_M10": "mucape",
        "MU_CAPE": "mucape",
        "ML_MIXR": "q",
        "SB_WMAX": "ws",
        "SB_LFC": "h",
        "ZeroHeight": "hzerocl",
        "ML_LCL": "cbh",
        "MW_13": "ws"
    }

    filename = datapath + "/" + basetime + "/ensO640_SFC_" + nmem + "_mucape_" + basetime + "_" + step + ".grib"
    template = mv.read(filename)
    ngrid = int(mv.count(mv.values(template[0])))

    predictors_all = {}
    nchunks = 2
    ncstep = int(ngrid / nchunks)
    nstart = 0
    for nchunk in range(0, nchunks):
        if nchunk == nchunks - 1:
            nstop = ngrid
        else:
            nstop = nstart + ncstep

        tracemalloc.start()

        data3d = {}
        data2d = {}
        for forecast_hour in all_timesteps:
            if forecast_hour == all_timesteps[0]:
                for fieldname in fieldnames_3d:
                    if (not fieldname in data3d):
                        data3d[fieldname] = {}

                    if (not str(forecast_hour) in data3d[fieldname]):
                        data3d[fieldname][forecast_hour] = {}

                    filename = datapath + "/" + basetime + "/ensO640_PL_" + nmem + "_" + fieldname + "_" + basetime + "_" + str(forecast_hour) + ".grib"
                    param_data = mv.read(filename)
                    plevel_list_all = mv.grib_get_long(param_data,'level')
                    plevel_list = mv.unique(plevel_list_all)
                    n_plevels = len(plevel_list)
                    n_members = int(len(plevel_list_all)/n_plevels)

                    param_values = mv.values(param_data)[:,nstart:nstop].copy()
                    n_grid = param_values.shape[1]
                    param_values = param_values.reshape(n_members, n_plevels, n_grid)
                    param_values = np.moveaxis(param_values, 1, 0)
                    data3d[fieldname][forecast_hour] = param_values

                    del param_values

                for fieldname in fieldnames_2d:
                    if (not fieldname in data2d):
                        data2d[fieldname] = {}

                    if (not str(forecast_hour) in data2d[fieldname]):
                        data2d[fieldname][forecast_hour] = {}

                    filename = datapath + "/" + basetime + "/ensO640_SFC_" + nmem + "_" + fieldname + "_" + basetime + "_" + str(forecast_hour) + ".grib"
                    param_data = mv.read(filename)
                    param_values = mv.values(param_data)[nstart:nstop].copy()
                    n_grid = param_values.shape[0]
                    n_members = int(len(param_data))
                    param_values = param_values.reshape(n_members, n_grid)
                    data2d[fieldname][forecast_hour] = param_values

                    del param_values

                print('Computing 2r from 2d and 2t')

                if (not '2r' in data2d):
                    data2d['2r'] = {}
                e  =  6.1121 * np.exp( (18.678 - ((data2d['2d'][forecast_hour][:,:] - 273.16)/234.5)) * ((data2d['2d'][forecast_hour][:,:] - 273.16) / (257.14 + (data2d['2d'][forecast_hour][:,:] - 273.16))) )
                es =  6.1121 * np.exp( (18.678 - ((data2d['2t'][forecast_hour][:,:] - 273.16)/234.5)) * ((data2d['2t'][forecast_hour][:,:] - 273.16) / (257.14 + (data2d['2t'][forecast_hour][:,:] - 273.16))) )
                data2d['2r'][forecast_hour] = 100 * e / es

            else:
                for fieldname in fieldnames_2d_add:
                    if (not fieldname in data2d):
                        data2d[fieldname] = {}

                    if (not str(forecast_hour) in data2d[fieldname]):
                        data2d[fieldname][forecast_hour] = {}

                    filename = datapath + "/" + basetime + "/ensO640_SFC_" + nmem + "_" + fieldname + "_" + basetime + "_" + str(
                        forecast_hour) + ".grib"
                    param_data = mv.read(filename)
                    param_values = mv.values(param_data)[nstart:nstop].copy()
                    n_grid = param_values.shape[0]
                    n_members = int(len(param_data))
                    param_values = param_values.reshape(n_members, n_grid)

                    data2d[fieldname][forecast_hour] = param_values

                    del param_values

                predictors = CalculateARCHaMoSPredictors(data2d, data3d, forecast_hour, plevel_list, all_timesteps)
                if len(predictors_all) == 0:
                    predictors_all = predictors.copy()
                else:
                    for k, v in predictors_all.items():
                        predictors_all[k] = np.hstack((v, predictors[k]))

        nstart = nstop
    SaveParametersAsGRIB(predictors_all, params_dict, template, forecast_hour)

def SaveParametersAsGRIB(params, params_dict, template, forecast_hour):
    prev_step = all_timesteps[all_timesteps.index(forecast_hour) - 1]
    print(prev_step, all_timesteps.index(forecast_hour))

    for key in params:
        template = mv.set_values(template, params[key])
        template = mv.grib_set(template,
                                    ["shortName", params_dict[key],
                                    "step", prev_step])

        mv.write(datapath + "/" + basetime + "/predictors/predictors_" + nmem + "_" + key + "_" + basetime + "_" + str(prev_step) + ".grib", template)

    return

def CalculateARCHaMoSPredictors(data2d, data3d, step, levels, all_timesteps):

    #   Required parameters for various models:
    #
    #   Lightning model: MU_LI, meanRH_500-850, mcpr, lsm, MU_MIXR
    #   Hail 2 cm: BS_EFF_MU, MU_CAPE_M10, ML_MIXR, ZeroHeight
    #   Hail 5 cm: BS_EFF_MU, MU_CAPE_M10, ML_LCL, ML_MixingRatio
    #
    #   All: # MU_LI, meanRH_500-850, mcpr, lsm, MU_MIXR, BS_EFF_MU, MU_CAPE_M10, ML_MixingRatio, ZeroHeight, ML_LCL

    print('Calculating predictors...')
    params = {}
    prev_step = all_timesteps[all_timesteps.index(step) - 1]

    print('Calculating RHmean...')
    params['RHmean'] = FindMean(data3d['r'][prev_step][:, :, :], 500, 850, levels)

    #  Calculation MW13

    print('Calculating Mean Wind between 1 and 3 km AGL')
    zsfc = data2d['z'][step]/9.81

    # approximation here is to take the u and v components at 1, 2 and 3 km AGL and average them to get a speed

    u1 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['u'][prev_step][:, :, :], 1000 + zsfc)
    v1 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['v'][prev_step][:, :, :], 1000 + zsfc)
    u2 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['u'][prev_step][:, :, :], 2000 + zsfc)
    v2 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['v'][prev_step][:, :, :], 2000 + zsfc)
    u3 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['u'][prev_step][:, :, :], 3000 + zsfc)
    v3 = InterpolateToHeight2D(data3d['z'][prev_step][:, :, :], data3d['v'][prev_step][:, :, :], 3000 + zsfc)

    mean_u = (u1 + u2 + u3) / 3.0
    mean_v = (v1 + v2 + v3) / 3.0

    MW_13 = np.sqrt( mean_u**2 + mean_v**2 )

    print('Calculating theta-ep...')
    p = np.broadcast_to(np.asarray(levels)[:, None, None], data3d['t'][prev_step][:, :, :].shape)
    mixr = data3d['q'][prev_step][:, :, :] / (1 - data3d['q'][prev_step][:, :, :])
    theta_ep = Theta_ep(data3d['t'][prev_step][:, :, :], p, mixr)

    print('Finding most unstable parcel...')
    psfc = data2d['sp'][prev_step][:, :]
    
    theta_ep[p > psfc/100 - 50] = -9999
    level_max_theta_ep = np.argmax(theta_ep[levels.index(700):levels.index(1000) + 1, :, :], axis=0) + levels.index(700)
    level_max_theta_ep = level_max_theta_ep.flatten()

    nz, nx, ny = p.shape[0], p.shape[1], p.shape[2]

    p = np.transpose(np.reshape(p, (nz, nx * ny)) )
    t = np.transpose(np.reshape(data3d['t'][prev_step][:, :, :], (nz, nx * ny)))
    u = np.transpose(np.reshape(data3d['u'][prev_step][:, :, :], (nz, nx * ny)))
    v = np.transpose(np.reshape(data3d['v'][prev_step][:, :, :], (nz, nx * ny)))
    q = np.transpose(np.reshape(data3d['q'][prev_step][:, :, :], (nz, nx * ny)))
    z = np.transpose(np.reshape(data3d['z'][prev_step][:, :, :], (nz, nx * ny))) / 9.81
    theta_ep = np.transpose(np.reshape(theta_ep, (nz, nx * ny)))

    z_maxthep = z[range(len(level_max_theta_ep)), level_max_theta_ep]
    q_maxthep = q[range(len(level_max_theta_ep)), level_max_theta_ep]
    p_maxthep = p[range(len(level_max_theta_ep)), level_max_theta_ep]
    t_maxthep = t[range(len(level_max_theta_ep)), level_max_theta_ep]
    theta_ep_maxthep = theta_ep[range(len(level_max_theta_ep)), level_max_theta_ep]

    w_maxthep = (q_maxthep / (1 - q_maxthep))
    e_maxthep = (w_maxthep * p_maxthep) / (w_maxthep + eps)
    T_LCL = (2840 / (3.5 * np.log(t_maxthep) - np.log(e_maxthep) - 4.805)) + 55  # Bolton's approximation from Emanuel, 1996 (Eq. 4.6.24)
    p_LCL = p_maxthep * np.power((T_LCL / t_maxthep), cpd / rd)
    z_LCL = (t_maxthep - T_LCL) * 125 + z_maxthep                               # This is a rough approximation

    print('max z_LCL: ', z_LCL.max(), z_LCL.min())

    diff = t_maxthep - T_LCL
    print(data2d['z'][step]/9.81)
    print(zsfc.min(), zsfc.max())
    print(diff.min(), diff.max())
    print('min, max e_maxthep: ', e_maxthep.min(), e_maxthep.max())
    print('e_maxthep: ', e_maxthep)
    lcl = z_LCL - data2d['z'][step] / 9.81
    print('LCL: ', lcl)
    print('min,max LCL: ', lcl.min(), lcl.max())
    print('LCL = 0: ', (lcl <=10).sum())

    params['MU_MIXR']      = np.reshape(np.transpose(q_maxthep), (nx, ny)) * 1000
    MU_LI                  = np.reshape(CalcLI(theta_ep_maxthep, t[:, levels.index(500)], q[:, levels.index(500)], 500), (nx, ny))
    params['mcpr']         = (data2d['cp'][step][:, :] - data2d['cp'][prev_step][:, :]) * 1000 # / (3600 * (step - prev_step))
    params['lsm']          = data2d['lsm'][prev_step][:, :]
    params['ZeroHeight']   = data2d['deg0l'][prev_step][:, :]

    EL, MU_CAPE, MU_CAPE_M10, MU_CIN, MU_LFC = CalcCAPE(theta_ep_maxthep, t[:, :], q[:, :], z[:, :], p, p_LCL)
    MU_EFF_BS = CalcBS_EFF(z_maxthep, EL, u, v, z)
    params['MU_EFF_BS']    = np.reshape(MU_EFF_BS, (nx, ny))
    params['MU_CAPE_M10']  = np.reshape(MU_CAPE_M10, (nx, ny))
    MU_CAPE[MU_CAPE < -1]  = np.nan
    params['MU_CAPE']      = np.reshape(MU_CAPE, (nx, ny))
    params['ML_MIXR']      = params['MU_MIXR']                                               # This is an approximation
    params['ML_LCL']       = np.reshape(z_LCL, (nx, ny))  - data2d['z'][step] / 9.81         # This is an approximation
    MU_LI[np.isnan(MU_LI)] = 15                                                              
    params['MU_LI']        = MU_LI
    params['MW_13']        = MW_13

    print('z LCL :', params['ML_LCL'].min(),params['ML_LCL'].max() )

    t_sfc = data2d['2t'][prev_step][:, :]
    r_sfc = data2d['2r'][prev_step][:, :]
    p_sfc = data2d['sp'][prev_step][:, :]
    e_sfc = r_sfc * 6.1121 * np.exp(
        (18.678 - ((t_sfc - 273.16) / 234.5)) * ((t_sfc - 273.16) / (257.14 + (t_sfc - 273.16))))
    q_sfc = 0.622 * e_sfc / (p_sfc - e_sfc)
    T_LCL_sfc = (2840 / (3.5 * np.log(t_sfc) - np.log(
        e_sfc) - 4.805)) + 55  # Bolton's approximation from Emanuel, 1996 (Eq. 4.6.24)
    p_LCL_sfc = (p_sfc * np.power((T_LCL_sfc / t_sfc), cpd / rd)).flatten()

    theta_ep_sfc = Theta_ep(t_sfc, p_sfc / 100, q_sfc).flatten()

    EL, SB_CAPE, SB_CAPE_M10, SB_CIN, SB_LFC = CalcCAPE(theta_ep_sfc, t[:, :], q[:, :], z[:, :], p, p_LCL_sfc)

    # params['SB_CAPE_M10'] = SB_CAPE_M10
    params['SB_WMAX'] = np.reshape(np.sqrt(2 * SB_CAPE), (nx, ny))
    params['SB_LFC'] = np.reshape(SB_LFC, (nx, ny))

    return params

def CalcBS_EFF(z_maxthep, EL, u, v, z):

    # Calculates effective bulk shear
    bottom_height = z_maxthep
    top_height = (EL - z_maxthep) / 2 + z_maxthep

    u_bottom = InterpolateToHeight2D(z, u, bottom_height)
    v_bottom = InterpolateToHeight2D(z, v, bottom_height)
    u_top = InterpolateToHeight2D(z, u, top_height)
    v_top = InterpolateToHeight2D(z, v, top_height)

    BS_EFF = np.sqrt((u_top - u_bottom) * (u_top - u_bottom) + (v_top - v_bottom) * (v_top - v_bottom))
    BS_EFF[EL == 0] = 0

    return BS_EFF


def CalcLI(t_ep_level, t_env, q_env, p_req):

    lookuptable = xr.open_dataset(parcel_lookupfile)

    f = RGI((lookuptable.theta_ep, lookuptable.p), lookuptable.Tparcel.values, \
            method='linear', bounds_error=False, fill_value=np.nan)

    t_parcel = f((t_ep_level, p_req))
    r_parcel = eps * (es(t_parcel) / (p_req - es(t_parcel)))
    tv_parcel = t_parcel * (1 + 0.61 * r_parcel)

    # print('r_parcel = ', r_parcel)
    # print('tv_parcel = ', tv_parcel)

    r_env = q_env / (1 - q_env)
    tv_env = t_env * (1 + 0.61 * r_env)

    # print('r_env = ', r_env)
    # print('tv_env = , ', tv_env)

    LI = tv_env - tv_parcel

    return LI


def CalcCAPE(t_ep_level, t_env, q_env, z_env, p_levels, p_LCL):

    lookuptable = xr.open_dataset(parcel_lookupfile)

    f = RGI((lookuptable.theta_ep, lookuptable.p), lookuptable.Tparcel.values, \
            method='linear', bounds_error=False, fill_value=np.nan)

    t_ep_level = np.broadcast_to(t_ep_level[:, None], p_levels.shape)
    t_parcel = f((t_ep_level, p_levels))
    r_parcel = eps * (es(t_parcel) / (p_levels - es(t_parcel)))
    tv_parcel = t_parcel * (1 + 0.61 * r_parcel)

    r_env = q_env / (1 - q_env)
    tv_env = t_env * (1 + 0.61 * r_env)

    buoyancy = tv_parcel - tv_env
    neg_buoyancy = - np.copy(buoyancy)

    buoyancy[p_levels > p_LCL[:, None]] = 0
    neg_buoyancy[p_levels > p_LCL[:, None]] = 0

    buoyancy[buoyancy < 0] = 0
    neg_buoyancy[neg_buoyancy < 0] = 0

    dz = - np.diff(z_env)
    dCAPE = ((buoyancy[:, :-1] + buoyancy[:, 1:]) / (t_env[:, 1:] + t_env[:, :-1])) * 9.81 * dz
    dCIN = ((neg_buoyancy[:, :-1] + neg_buoyancy[:, 1:]) / (t_env[:, 1:] + t_env[:, :-1])) * 9.81 * dz

    dCAPE[dCAPE < 0] = 0

    dCAPE_M10 = np.copy(dCAPE)
    dCAPE_M10[(t_env[:, 1:] + t_env[:, :-1]) / 2 > 263.15] = 0
    EL_level = np.argmax(dCAPE > 0, axis=1)
    EL = z_env[range(len(EL_level)), EL_level]
    LFC_level = np.argmax(np.logical_and(dCIN[:, 1:] > 0, dCAPE[:, :-1] > 0), axis=1)
    LFC = z_env[range(len(LFC_level)), LFC_level]

    CAPE = np.nansum(dCAPE, axis=1)
    dCIN[LFC_level[:, None] > np.arange(dCIN.shape[1])] = 0

    CIN = np.nansum(dCIN, axis=1)
    CIN[LFC_level == 0] = 0

    CAPE_M10 = np.nansum(dCAPE_M10, axis=1)
    EL[CAPE == 0] = 0

    return EL, CAPE, CAPE_M10, CIN, LFC


def es(T):  # saturation mixing ratio

    # es in hPa/mb
    es = 6.1121 * np.exp((18.678 - ((T - 273.16) / 234.5)) * ((T - 273.16) / (257.14 + (T - 273.16))))  # Buck equation

    return es



def FindMean(param, min_lev, max_lev, levels):
    weights = np.zeros_like(param)
    my_levels = np.broadcast_to(np.asarray(levels)[:, None, None], weights.shape)

    # simple not very accurate option:
    weights[np.logical_and(my_levels > min_lev, my_levels < max_lev)] = 1
    my_mean = np.sum(param * weights, axis=0) / np.sum(weights, axis=0)

    return my_mean


def Theta_ep(T, p, r):
    # theta_ep -> pseudoequivalent potential temperature
    #    print('SHAPES: ', T.shape, p.shape, r.shape)
    e = (r * p) / (r + eps)
    T_LCL = (2840 / (
                3.5 * np.log(T) - np.log(e) - 4.805)) + 55  # Bolton's approximation from Emanuel, 1996 (Eq. 4.6.24)
    theta_ep = T * ((1000 / p) ** (0.2854 * (1 - 0.28 * r))) * np.exp(
        r * (1 + 0.81 * r) * ((3376 / T_LCL) - 2.54))  # Bolton's approximation from Emanuel, 1996 (Eq. 4.7.9)

    return theta_ep






def InterpolateToHeight2D(z, param, height):

    nlevs = z.shape[1]

#    print('HEIGHT SHAPE', height.shape)
#    print('Z SHAPE', z.shape)

    # returns index of first level (from lower to higher pressure for which z < height)
    index_bottom = np.argmax(z < height[:, None], axis=1)
    index_top = index_bottom - 1

    index_bottom[index_bottom == nlevs - 1] = nlevs - 1# where not found, set both bottom and top to lowest
    index_top[index_bottom == nlevs - 1] = nlevs - 2

    index_bottom[index_top < 0] = nlevs - 1## where not found, set both bottom and top to lowest
    index_top[index_top < 0] = nlevs - 2#

#    print('index_top', index_top.shape, np.max(index_top), np.nanmax(index_top), np.min(index_top), np.nanmin(index_top))
#    print('index_bottom', index_bottom.shape, np.max(index_bottom), np.nanmax(index_bottom), np.min(index_bottom), np.nanmin(index_bottom))

    z_under = np.choose(index_bottom, np.transpose(z))
    z_over  = np.choose(index_top, np.transpose(z))

#    print('param', param.shape, np.max(param), np.nanmax(param), np.min(param), np.nanmin(param))

    param_under = np.choose(index_bottom, np.transpose(param))
    param_over  = np.choose(index_top, np.transpose(param))

    quotient = (height - z_under) / (z_over - z_under)

    myparam = quotient * param_over + (1 - quotient) * param_under

    myparam[index_top == z.shape[0] - 1] = np.nan

    return myparam


def MMM(name, param):

    print(name, param.shape, np.min(param), np.mean(param), np.max(param))
    return


if __name__ == "__main__":
    main()

print('Memory used: ', tracemalloc.get_traced_memory())
tracemalloc.stop()