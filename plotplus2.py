import os, functools
import numpy as np
import mpkit.gpf as gpf
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as ciosr
import matplotlib.pyplot as plt
import scipy.ndimage as snd
from datetime import datetime, timedelta

__version__ = '0.1.0'

_ShapeFileDir = os.path.join(os.path.split(__file__)[0], 'shapefile')
_ProvinceDir = os.path.join(_ShapeFileDir, 'CP/ChinaProvince')
_CityDir = os.path.join(_ShapeFileDir, 'CHN/CHN_adm2')
_CityTWDir = os.path.join(_ShapeFileDir, 'TWN/TWN_adm2')
_CountyDir = os.path.join(_ShapeFileDir, 'CHN/CHN_adm3')

_gray = '#222222'
_projshort = dict(P='PlateCarree', L='LambertConformal', M='Mercator',
    N='NorthPolarStereo', G='Geostationary')

class PlotError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class Plot:

    def __init__(self, figsize=(7,5), dpi=180):
        self.mmnote = ''
        self.family = 'Segoe UI Emoji'
        self.dpi = dpi
        self.fig = plt.figure(figsize=figsize)
        self.ax = None
        self.fontsize = dict(title=8, timestamp=5, mmnote=5, clabel=5, 
            cbar=5, gridvalue=5, mmfilter=6, parameri=4, legend=6)
        self.linecolor = dict(coastline=_gray, country=_gray,
            province=_gray, city=_gray, county=_gray, parameri='k')
        self.linewidth = dict(coastline=0.3, country=0.3, province=0.2,
            city=0.1, county=0.1, parameri=None)

    def setfamily(self, f):
        self.family = f

    def setfontsize(self, name, size):
        self.fontsize[name] = size

    def setlinecolor(self, name, color):
        self.linecolor[name] = color

    def setlinewidth(self, name, width):
        self.linewidth[name] = width

    def setdpi(self, d):
        self.dpi = d

    def setxy(self, georange, res):
        self.latmin, self.latmax, self.lonmin, self.lonmax = tuple(georange)
        self.x = np.arange(self.lonmin, self.lonmax+res, res)
        self.y = np.arange(self.latmin, self.latmax+res, res)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.res = res

    def setmap(self, key=None, **kwargs):
        '''key: map projection (refer to cartopy.crs) or built-in map name
        projection shortkey: P - PlateCarree / L - LambertConformal /
            N - NorthPolarStereo / M - Mercator / G - Geostationary
        built-in map name: chinaproper / chinamerc / chinalambert / europe /
            euroasia / northamerica / northpole
        for NorthPolarStereo, boundinglat is accepted'''
        if key == 'chinaproper':
            key = 'P'
            kwargs.update(georange=(20,40,100,130))
        elif key == 'chinamerc':
            key = 'M'
            kwargs.update(georange=(15,50,72.5,135))
        elif key == 'chinalambert':
            key = 'L'
            kwargs.update(georange=(15,55,80,125), central_longitude=102.5,
                central_latitude=40, standard_parallels=(40,40))
        elif key == 'euroasia':
            key = 'L'
            kwargs.update(georange=(5,75,55,145), central_longitude=100,
                central_latitude=40, standard_parallels=(40,40))
        elif key == 'europe':
            key = 'L'
            kwargs.update(georange=(5,75,-25,65), central_longitude=20,
                central_latitude=40, standard_parallels=(40,40))
        elif key == 'northamerica':
            key = 'L'
            kwargs.update(georange=(5,75,-145,-55), central_longitude=-100,
                central_latitude=40, standard_parallels=(40,40))
        elif key == 'northpole':
            key = 'N'
            kwargs.update(georange=(15,90,-180,180), central_longitude=105)
        key = _projshort.get(key.upper(), key)
        if 'georange' in kwargs:
            georange = kwargs.pop('georange')
        if key == 'NorthPolarStereo' and 'boundinglat' in kwargs:
            georange = (kwargs.pop('boundinglat'), 90, -180, 180)
        self.proj = key
        self.ax = plt.axes(projection=getattr(ccrs, key)(**kwargs))
        extent = georange[2:] + georange[:2]
        self.ax.set_extent(extent)

    def usemap(self, proj, extent=None):
        self.ax = plt.axes(projection=proj)
        self.proj = type(proj).__name__
        if extent:
            self.ax.set_extent(extent)

    def drawcoastline(self, lw=None, color=None, res='50m'):
        lw = self.linewidth['coastline'] if lw is None else lw
        color = self.linecolor['coastline'] if color is None else color
        self.ax.add_feature(self.getfeature('physical', 'coastline', res,
            facecolor='none', edgecolor=color), linewidth=lw)
    
    def drawcountry(self, lw=None, color=None, res='50m'):
        lw = self.linewidth['country'] if lw is None else lw
        color = self.linecolor['country'] if color is None else color
        self.ax.add_feature(self.getfeature('cultural', 'admin_0_boundary_lines_land',
            res, facecolor='none', edgecolor=color), linewidth=lw)

    @functools.lru_cache(maxsize=32)
    def getfeature(self, *args, **kwargs):
        return cfeature.NaturalEarthFeature(*args, **kwargs)
    
    def drawprovince(self, lw=None, color=None):
        lw = self.linewidth['province'] if lw is None else lw
        color = self.linecolor['province'] if color is None else color
        self.ax.add_feature(cfeature.ShapelyFeature(ciosr.Reader(_ProvinceDir).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor=color), linewidth=lw)
    
    def drawcity(self, lw=None, color=None):
        lw = self.linewidth['city'] if lw is None else lw
        color = self.linecolor['city'] if color is None else color
        self.ax.add_feature(cfeature.ShapelyFeature(ciosr.Reader(_CityDir).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor=color), linewidth=lw)
        self.ax.add_feature(cfeature.ShapelyFeature(ciosr.Reader(_CityTWDir).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor=color), linewidth=lw)

    def drawcounty(self, lw=None, color=None):
        lw = self.linewidth['county'] if lw is None else lw
        color = self.linecolor['county'] if color is None else color
        self.ax.add_feature(cfeature.ShapelyFeature(ciosr.Reader(_CountyDir).geometries(),
            ccrs.PlateCarree(), facecolor='none', edgecolor=color), linewidth=lw)

    def drawparameri(self, lw=None, color=None, fontsize=None, **kwargs):
        import cartopy.mpl.gridliner as cmgl
        import matplotlib.ticker as mticker
        lw = self.linewidth['parameri'] if lw is None else lw
        color = self.linecolor['parameri'] if color is None else color
        fontsize = self.fontsize['parameri'] if fontsize is None else fontsize
        gl = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=lw,
            color=color, linestyle='- -')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(-80, 81, 10))
        gl.xformatter = cmgl.LONGITUDE_FORMATTER
        gl.yformatter = cmgl.LATITUDE_FORMATTER
        gl.xlabel_style = dict(size=fontsize, color=color, family=self.family)
        gl.ylabel_style = dict(size=fontsize, color=color, family=self.family)
        if self.proj == 'PlateCarree' or self.proj == 'Mercator':
            gl.xlines = False
            gl.ylines = False
    
    def draw(self, cmd):
        cmd = cmd.lower()
        if ' ' in cmd:
            for c in cmd.split():
                self.draw(c)
        else:
            if cmd == 'coastline' or cmd == 'coastlines':
                self.drawcoastline()
            elif cmd == 'country' or cmd == 'countries':
                self.drawcountry()
            elif cmd == 'province' or cmd == 'provinces':
                self.drawprovince()
            elif cmd == 'city' or cmd == 'cities':
                self.drawcity()
            elif cmd == 'county' or cmd == 'counties':
                self.drawcounty()
            elif cmd == 'parameri' or cmd == 'meripara':
                self.drawparameri()
            else:
                print('Illegal draw command: %s' % (cmd))

    def interpolation(self, data, ip=1):
        if ip <= 1:
            return self.x, self.y, data
        else:
            nx = np.arange(self.lonmin, self.lonmax+self.res/ip, self.res/ip)
            ny = np.arange(self.latmin, self.latmax+self.res/ip, self.res/ip)
            newx, newy = np.meshgrid(nx, ny)
            xcoords = (len(self.x)-1)*(newx-self.x[0])/(self.x[-1]-self.x[0])
            ycoords = (len(self.y)-1)*(newy-self.y[0])/(self.y[-1]-self.y[0])
            coords = [ycoords, xcoords]
            ndata = snd.map_coordinates(data, coords, order=3, mode='nearest')
            return nx, ny, ndata

    def stepcal(self, num, ip=1):
        totalpt = (self.lonmax - self.lonmin) / self.res * ip
        return int(totalpt / num)

    def legend(self, lw=0., **kwargs):
        rc = dict(loc='upper right', framealpha=0.)
        rc.update(kwargs)
        ret = self.ax.legend(prop=dict(family=self.family, size=self.fontsize['legend']),
                             **rc)
        ret.get_frame().set_linewidth(lw)
        return ret
    
    def plot(self, *args, **kwargs):
        ret = self.ax.plot(*args, **kwargs)
        return ret

    def contour(self, data, clabel=True, clabeldict=dict(), ip=1, color='k', lw=0.5,
                filter=None, **kwargs):
        x, y, data = self.interpolation(data, ip)
        kwargs.update(colors=color, linewidths=lw, transform=ccrs.PlateCarree())
        c = self.ax.contour(x, y, data, **kwargs)
        if clabel:
            if 'levels' in clabeldict:
                clabellevels = clabeldict.pop('levels')
            else:
                clabellevels = kwargs['levels']
            clabeldict.update(fmt='%d', fontsize=self.fontsize['clabel'])
            labels = self.ax.clabel(c, clabellevels, **clabeldict)
            for l in labels:
                l.set_family(self.family)
        if filter is not None:
            self.maxminfilter(data, res=self.res/ip, **filter)
        return c

    def contourf(self, data, gpfcmap=None, cbar=False, cbardict=dict(), ip=1,
                 filter=None, vline=None, vlinedict=dict(), **kwargs):
        if gpfcmap:
            kwargs = merge_dict(kwargs, gpf.cmap(gpfcmap))
        x, y, data = self.interpolation(data, ip)
        kwargs.update(transform=ccrs.PlateCarree())
        c = self.ax.contourf(x, y, data, **kwargs)
        if cbar:
            if 'ticks' not in cbardict:
                levels = kwargs['levels']
                step = len(levels) // 40 + 1
                cbardict.update(ticks=levels[::step])
            rc = dict(size='2%', pad='1%')
            merge_dict(cbardict, rc)
            if 'extend' in kwargs:
                cbardict.update(extend=kwargs.pop('extend'), extendfrac=0.02)
            self.colorbar(c, unit=kwargs.pop('unit', None), **cbardict)
        if filter is not None:
            self.maxminfilter(data, res=self.res/ip, **filter)
        if vline is not None:
            if 'color' not in vlinedict:
                vlinedict.update(colors='w')
            if 'lw' not in vlinedict:
                vlinedict.update(linewidths=0.6)
            else:
                vlinedict.update(linewidths=vlinedict.pop('lw'))
            self.ax.contour(x, y, data, levels=[vline], **vlinedict)
        return c

    def colorbar(self, mappable, unit=None, **kwargs):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if kwargs.pop('orientation', None) == 'horizontal':
            location = 'bottom'
            orientation = 'horizontal'
        else:
            location = 'right'
            orientation = 'vertical'
            self._colorbar_unit(unit)
        divider = make_axes_locatable(self.ax)
        import cartopy.mpl.geoaxes as cmga
        cax = divider.append_axes(location, size=kwargs.pop('size'), pad=kwargs.pop('pad'),
            map_projection=self.ax.projection)
        cb = self.fig.colorbar(mappable, orientation=orientation, cax=cax, **kwargs)
        self.fig.sca(self.ax)
        cb.ax.tick_params(labelsize=self.fontsize['cbar'], length=1.5)
        cb.outline.set_linewidth(0.3)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family(self.family)
        return cb

    def streamplot(self, u, v, color='w', lw=0.3, density=2, **kwargs):
        kwargs.update(color=color, linewidth=lw, density=density, transform=ccrs.PlateCarree())
        ret = self.ax.streamplot(self.x, self.y, u, v, **kwargs)
        return ret

    def barbs(self, u, v, color='k', lw=0.5, length=4, num=12, **kwargs):
        kwargs.update(color=color, linewidth=lw, length=length, transform=ccrs.PlateCarree(),
            regrid_shape=num)
        nh = np.meshgrid(self.x, self.y)[1] >= 0
        unh = np.ma.masked_where(~nh, u)
        vnh = np.ma.masked_where(~nh, v)
        ret = self.ax.barbs(self.x, self.y, unh, vnh, **kwargs)
        ush = np.ma.masked_where(nh, u)
        vsh = np.ma.masked_where(nh, v)
        retsh = self.ax.barbs(self.x, self.y, ush, vsh, flip_barb=True, **kwargs)
        return ret, retsh

    def quiver(self, u, v, num=40, scale=500, qkey=False, qkeydict=dict(), **kwargs):
        kwargs.update(width=0.0015, headwidth=3, scale=scale, transform=ccrs.PlateCarree(),
            regrid_shape=num)
        q = self.ax.quiver(self.x, self.y, u, v, **kwargs)
        if qkey:
            if 'x' in qkeydict and 'y' in qkeydict:
                x = qkeydict.pop('x')
                y = qkeydict.pop('y')
            else:
                x, y = 0.5, 1.01
            unit = 'm/s' if 'unit' not in qkeydict else qkeydict.pop('unit')
            self.ax.quiverkey(q, x, y, scale, '%d%s' % (scale, unit), labelpos='W',
                              fontproperties=dict(family=self.family, size=8))
        return q

    def gridvalue(self, data, num=20, fmt='{:d}', color='b', fontsize=None,
                  shadow=False, **kwargs):
        if fontsize is None:
            fontsize = self.fontsize['gridvalue']
        if shadow:
            import matplotlib.patheffects as mpatheffects
            kwargs.update(path_effects=[mpatheffects.withSimplePatchShadow(
                offset=(0.5,-0.5), alpha=0.7)])
        step = self.stepcal(num)
        kwargs.update(color=color, fontsize=fontsize, ha='center', va='center',
                      family=self.family, transform=ccrs.PlateCarree())
        if self.proj == 'PlateCarree':
            meri, para = len(self.y), len(self.x)
            for i in range(1, meri-1, step):
                for j in range(1, para-1, step):
                    self.ax.text(j*self.res+self.lonmin, i*self.res+self.latmin,
                                fmt.format(data[i][j]), **kwargs)
        else:
            x1, x2, y1, y2 = self.ax.get_extent()
            deltax, deltay = x2 - x1, y2 - y1
            x1 += 0.02 * deltax
            x2 -= 0.02 * deltax
            y1 += 0.02 * deltay
            y2 -= 0.02 * deltay
            x = np.linspace(x1, x2, num)
            y = np.linspace(y1, y2, num)
            xx, yy = np.meshgrid(x, y)
            points = ccrs.Geodetic().transform_points(self.ax.projection, xx, yy)
            points_round = np.round(points/self.res) * self.res
            values = data[np.searchsorted(self.x, points_round[:,:,0]), 
                np.searchsorted(self.y, points_round[:,:,1])]
            result = np.dstack((points[:,:,:2], values))
            for i in result:
                for j in i:
                    lon, lat, value = tuple(j)
                    self.ax.text(lon, lat, fmt.format(value), **kwargs)

    def maxminfilter(self, data, type='min', fmt='{:d}', weight='bold', color='b',
                     fontsize=None, window=15, vmin=-1e7, vmax=1e7, shadow=False,
                     **kwargs):
        '''Use res keyword or ip keyword to interpolate'''
        if fontsize is None:
            fontsize = self.fontsize['mmfilter']
        if shadow:
            import matplotlib.patheffects as mpatheffects
            kwargs.update(path_effects=[mpatheffects.withSimplePatchShadow(
                offset=(0.5,-0.5), alpha=0.7)])
        kwargs.update(fontweight=weight, color=color, fontsize=fontsize,
                      ha='center', va='center', transform=ccrs.PlateCarree())
        if 'res' in kwargs:
            res = kwargs.pop('res')
        elif 'ip' in kwargs:
            ip = kwargs.pop('ip')
            x, y, data = self.interpolation(data, ip)
            res = self.res / ip
        else:
            res = self.res
        if type == 'min':
            ftr = snd.minimum_filter
        elif type == 'max':
            ftr = snd.maximum_filter
        else:
            raise PlotError('Unsupported filter type!')
        dataftr = ftr(data, window, mode='reflect')
        yind, xind = np.where(data == dataftr)
        ymax, xmax = data.shape
        for y, x in zip(yind, xind):
            d = data[y, x]
            if d < vmax and d > vmin and x not in (0, xmax-1) and y not in (0, ymax-1):
                self.ax.text(x*res+self.lonmin, y*res+self.latmin, fmt.format(d), **kwargs)

    def title(self, s, nasdaq=True):
        if nasdaq:
            s = s + ' @NASDAQ'
        self.ax.text(0, 1.04, s, transform=self.ax.transAxes, fontsize=self.fontsize['title'],
                     family=self.family)

    def timestamp(self, basetime, fcsthour, duration=0, nearest=None):
        stdfmt = '%Y/%m/%d %HZ'
        if isinstance(basetime, str):
            basetime = datetime.strptime(basetime, '%Y%m%d%H')
        if duration:
            if duration > 0:
                fcsthour = fcsthour, fcsthour + duration
            else:
                fcsthour = fcsthour + duration, fcsthour
        elif nearest:
            validtime = basetime + timedelta(hours=fcsthour - 1)
            nearesttime = validtime.replace(hour=validtime.hour // nearest * nearest)
            fcsthour = fcsthour + nearesttime.hour - validtime.hour - 1, fcsthour
        if isinstance(fcsthour, int):
            validtime = basetime + timedelta(hours=fcsthour)
            s = '%s [+%dh] valid at %s' % (basetime.strftime(stdfmt), fcsthour,
                                              validtime.strftime(stdfmt))
        elif isinstance(fcsthour, str):
            if fcsthour == 'an':
                s = basetime.strftime(stdfmt)
            else:
                s = ''
        else:
            fcsthour = tuple(fcsthour)
            fromhour, tohour = fcsthour
            fromtime = basetime + timedelta(hours=fromhour)
            totime = basetime + timedelta(hours=tohour)
            s = '%s [+%d~%dh] valid from %s to %s' % (basetime.strftime(stdfmt),
                                                      fromhour, tohour, fromtime.strftime(stdfmt),
                                                      totime.strftime(stdfmt))
        self._timestamp(s)

    def _timestamp(self, s):
        self.ax.text(0, 1.01, s, transform=self.ax.transAxes,
                     fontsize=self.fontsize['timestamp'], family=self.family)

    def _colorbar_unit(self, s):
        if s:
            self.ax.text(1.05, 1.01, s, transform=self.ax.transAxes, ha='right',
                         fontsize=self.fontsize['timestamp'], family=self.family)

    def maxminnote(self, data, name, unit, type='max', fmt='%.1f'):
        type = type.lower()
        if type == 'max':
            typestr = 'Max.'
            notevalue = np.amax(data)
        elif type == 'min':
            typestr = 'Min.'
            notevalue = np.amin(data)
        elif type == 'mean':
            typestr = 'Mean'
            notevalue = np.mean(data)
        else:
            raise PlotError('Unsupported type!')
        notestr = '%s %s: ' % (typestr, name) + fmt % notevalue + ' ' + unit
        if self.mmnote != '':
            self.mmnote = self.mmnote + ' | ' + notestr
        else:
            self.mmnote = notestr

    def _maxminnote(self, s):
        self.mmnote = s

    def stdsave(self, directory, basetime, fcsthour, imcode):
        if isinstance(basetime, str):
            basetime = datetime.strptime(basetime, '%Y%m%d%H')
        path = os.path.join(directory, '%s_%s_%d_%d%d%d%d.png' % \
                            (basetime.strftime('%Y%m%d%H'), imcode, fcsthour, self.latmin,
                             self.latmax, self.lonmin, self.lonmax))
        self.save(path)

    def save(self, path):
        self.ax.text(1, 1.01, self.mmnote, ha='right', transform=self.ax.transAxes,
                     fontsize=self.fontsize['mmnote'], family=self.family)
        self.ax.axis('off')
        self.fig.savefig(path, dpi=self.dpi, bbox_inches='tight', edgecolor='none',
                         pad_inches=0.05)

def merge_dict(a, b):
    '''Merge B into A without overwriting A'''
    for k, v in b.items():
        if k not in a:
            a[k] = v
    return a
