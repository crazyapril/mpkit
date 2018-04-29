from urllib.request import urlretrieve, ProxyHandler, build_opener, install_opener
from urllib.error import HTTPError
from ftplib import FTP
from os import path, remove, system
import numpy as np
import sys
from datetime import datetime, timedelta

_DATADIR_ = path.join(path.dirname(path.abspath(sys.argv[0])), 'data')
_GFSSTEP_ = 0.25
_ECMWFSTEP_ = 0.5
_ERASTEP_ = 0.75

_common_used_level_ = {'2m':'2_m_above_ground', '10m':'10_m_above_ground',
                    'msl':'mean_sea_level', '10mb':'10_mb', '200mb':'200_mb',
                       '500mb':'500_mb', '700mb':'700_mb', '850mb':'850_mb',
                       '925mb':'925_mb', 'eas':'entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29'}
_timeseq_ = 'AEIKMOQSWYT'
_productdict_ = {'500h':('H', '50', 'gh_500hPa'), 'mslp':('P', '89', 'msl'), '850t':('T', '85', 't_850hPa'),
                 '850u':('U', '85', 'u_850hPa'), '850v':('V', '85', 'v_850hPa'), '850w':('W', '85', 'ws_850hPa')}
_ecensdict_ = {'deter':('ECMF', ''), 'emean':('ECEM', 'em_'), 'esdev':('ECED', 'es_')}
_eradict_ = {'z':129, 't':130, 'u':131, 'v':132, 'q':133, 'sp':134, 'sd':141, 'sf':144, 'msl':151, 'r':157,
             'tcc':164, 'lsm':172, 'lsp':142, 'cp':143, 'ws':10, '2d':168, '2t':167, '10u':165, '10v':166,
             'tp':228, 'sst':34}

def installproxy():
    print('Proxy opened. IP Address: 202.112.26.250')
    proxy_support = ProxyHandler({'http': '202.112.26.250:8080'})
    opener = build_opener(proxy_support)
    install_opener(opener)

def GFS(level, parameter, basetime, fcsthour, georange, filename=None):
    '''level: surface, msl, 10m, 850mb, tropopause, etc
    parameter: TMP, HGT, PRES, UGRD, LAND, ACPCP, etc
    basetime: 2015050106 i.e.
    fcsthour: 0:240:3, 240:384:12
    georange: (latmin, latmax, lonmin, lonmax)
    filename: path of file to store the data'''
    latmin, latmax, lonmin, lonmax = georange
    if filename == None:
        filename = path.join(_DATADIR_, '%s_%s_%s_%d%d%d%d_%d.grib2' % (basetime, level,
                                                                        parameter, latmin, latmax, lonmin, lonmax, fcsthour))
    if path.isfile(filename.replace('grib2', 'bin')):
        print('File found.')
        return filename.replace('grib2', 'bin')
    elif path.isfile(filename):
        print('File found.')
        return filename
    try:
        level = _common_used_level_[level]
    except KeyError:
        pass
    if fcsthour % 3 != 0:
        url_prefix = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?'
    else:
        url_prefix = 'http://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
    url_fcst = 'file=gfs.t%sz.pgrb2.0p25.f%03d' % (basetime[-2:], fcsthour)
    url_rest = '&dir=%%2Fgfs.%s' % (basetime)
    url_parameter = '&var_%s=on' % (parameter)
    url_lev = '&lev_%s=on' % (level)
    url_georange = '&subregion=&leftlon=%d&rightlon=%d&toplat=%d&bottomlat=%d' % (lonmin, lonmax, latmax, latmin)
    url = url_prefix + url_fcst + url_lev + url_parameter + url_georange + url_rest
    try:
        urlretrieve(url, filename)
    except HTTPError as err:
        print(err)
        return
    return filename

def ECMWF(product, basetime, fcsthour, filename=None, mode='deter'):
    '''product: 500h, mslp, 850t, 850u, 850v, 850w
    basetime: 2015050106 i.e.
    fcsthour: 0:240:24
    filename: path of file to store the data
    mode: deter/emean/esdev'''
    try:
        parameter, level, suffix = _productdict_[product]
    except KeyError:
        print('Unrecognized product name. (500h/mslp/850t/850u/850v/850w)')
        return
    try:
        indicator, adder = _ecensdict_[mode]
    except KeyError:
        print('Unrecognized product name. (deter/emean/esdev)')
        return
    if mode != 'deter':
        product = product + '_' + mode
    if filename == None:
        filename = path.join(_DATADIR_, '%s_%s_%s.grib2' % (basetime, product, fcsthour))
    if path.isfile(filename.replace('grib2', 'bin')):
        print('File found.')
        return filename.replace('grib2', 'bin')
    elif path.isfile(filename):
        print('File found.')
        return filename
    if fcsthour == 0:
        fcststr = 'an'
    else:
        fcststr = '%dh' % (fcsthour)
    ftp = FTP()
    try:
        ftp.connect('data-portal.ecmwf.int', 21)
        ftp.login('wmo', 'essential')
        ftp.cwd(basetime + '0000')
    except Exception as err:
        print('Connect failed.', err)
        return
    ftpfile = 'A_H%sX%s%s%s%s_C_ECMF_%s_%s_%s%s_global_0p5deg_grib2.bin' % (parameter, _timeseq_[fcsthour//24], level,
                                                                            indicator, basetime[6:]+'00', basetime+'0000', fcststr, adder, suffix)
    try:
        ftp.retrbinary('RETR ' + ftpfile, open(filename, 'wb').write)
    except Exception as err:
        print('Download failed.', err)
        return
    #ftp.quit()
    return filename

def ERA(level, parameter, basetime, filename=None):
    '''level: sfc, 500mb, 700mb, 850mb, 925mb, 1000mb etc
    parameter: z, t, u, v etc
    basetime: 2015050106 i.e.
    filename: path of file to store the data'''
    try:
        paramint = _eradict_[parameter]
    except KeyError:
        print('Unrecognized parameter. Allowed:', _eradict_.keys())
        return
    if filename == None:
        filename = path.join(_DATADIR_, '%s_%s_%s.nc' % (basetime, level, parameter))
    if path.isfile(filename):
        print('File found.')
        return filename
    from ecmwfapi import ECMWFDataServer
    baseinfo = dict(dataset='interim', step='0', target=filename, grid='0.75/0.75',
                    stream='oper', type='an', param='{:d}.128'.format(paramint),
                    date='{:s}-{:s}-{:s}'.format(basetime[:4], basetime[4:6], basetime[6:8]),
                    time='{:s}:00:00'.format(basetime[-2:]), format='netcdf')
    baseinfo['class'] = 'ei'
    if level == 'sfc':
        baseinfo.update(levtype='sfc')
    elif level.endswith('mb'):
        baseinfo.update(levtype='pl', levelist=level[:-2])
    else:
        print('Unrecognized level.')
        return
    server = ECMWFDataServer()
    server.retrieve(baseinfo)
    return filename

def decoder(filename, filetype='bin', outname=None):
    '''filename: grib2 file name to decode
    filetype: bin/text/ieee/netcdf
    outname: path of decoded data file'''
    if filename == '' or not path.isfile(filename):
        return
    if filetype not in ('bin', 'text', 'ieee', 'netcdf'):
        print('Wrong filetype.')
        return
    if filename.endswith('.bin'):
        return filename
    if outname == None:
        if filetype == 'text':
            outname = filename.replace('.grib2', 'txt')
        elif filetype == 'netcdf':
            outname = filename.replace('.grib2', 'nc')
        else:
            outname = filename.replace('.grib2', '.'+filetype)
    system('wgrib2 %s -s -%s %s' % (filename, filetype, outname))
    #system('pause')
    return outname

def ModelData(model, basetime, fcsthour, georange, **kwargs):
    '''model: G/GFS, E/EC/ECMWF, ERA/INTERIM
    basetime: 2015050106 i.e.
    fcsthour: 0:240:1+240:384:12(GFS), 0:240:24(ECMWF), ignored(ERA)
    georange: (latmin, latmax, lonmin, lonmax)
    **kwargs: level parameter(GFS/ERA), product(ECMWF)'''
    latmin, latmax, lonmin, lonmax = georange
    if lonmax < 0:
        lonmax = 360 - lonmax
    if latmax < latmin or lonmax < lonmin:
        print('Illegal georange.')
        return
    if model.upper() in ('G', 'GFS'):
        outname = decoder(GFS(kwargs['level'], kwargs['parameter'], basetime, fcsthour, (latmin, latmax, lonmin, lonmax)))
        nx, ny = int((lonmax - lonmin) / _GFSSTEP_) + 1, int((latmax - latmin) / _GFSSTEP_) + 1
        if lonmin == 0 and lonmax == 360:
            nx -= 1
        if outname == '':
            print('No output.')
            return
        return np.fromfile(outname, dtype='f4')[1:-1].reshape(ny, nx)
    elif model.upper() in ('E', 'EC', 'ECMWF'):
        if 'mode' not in kwargs:
            kwargs.update(mode='deter')
        outname = decoder(ECMWF(kwargs['product'], basetime, fcsthour, mode=kwargs['mode']))
        if outname == '':
            print('No output')
            return
        data = np.fromfile(outname, dtype='f4')[1:-1].reshape(361, 720)
        if lonmin < 0:
            data = np.roll(data, 360, axis=1)
            lon0 = -180
        else:
            lon0 = 0
        sx, ex = int((lonmin - lon0) / _ECMWFSTEP_), int((lonmax - lon0) / _ECMWFSTEP_) + 1
        sy, ey = int((latmin + 90) / _ECMWFSTEP_), int((latmax + 90) / _ECMWFSTEP_) + 1
        return data[sy:ey, sx:ex]
    elif model.upper() in ('ERA', 'INTERIM'):
        outname = ERA(kwargs['level'], kwargs['parameter'], basetime)
        if outname == '':
            print('No output')
            return
        from netCDF4 import Dataset
        file = Dataset(outname)
        time = file.variables['time'][:]
        dt = datetime.strptime(basetime, '%Y%m%d%H')
        hourcal = (dt-datetime(1900,1,1))//timedelta(hours=1)
        ind = np.where(time == hourcal)[0]
        data = file.variables[kwargs['parameter']][ind[0],:,:]
        sx, ex = int((lonmin - 0) / _ERASTEP_), int((lonmax - 0) / _ERASTEP_) + 2
        sy, ey = int((latmin + 90) / _ERASTEP_), int((latmax + 90) / _ERASTEP_) + 2
        return data[sy:ey, sx:ex]
    else:
        print('Unknown model: %s. GFS/ECMWF/ERA supported.' % (model.upper()))
