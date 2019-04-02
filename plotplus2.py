import functools
import mpkit.gpf as gpf
import os
import warnings
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as ciosr
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as snd

__version__ = '0.1.0'

_ShapeFileDir = os.path.join(os.path.split(__file__)[0], 'shapefile')
_ProvinceDir = os.path.join(_ShapeFileDir, 'CP/ChinaProvince')
_CityDir = os.path.join(_ShapeFileDir, 'CHN/CHN_adm2')
_CityTWDir = os.path.join(_ShapeFileDir, 'TWN/TWN_adm2')
_CountyDir = os.path.join(_ShapeFileDir, 'CHN/CHN_adm3')

_gray = '#222222'
_projshort = dict(P='PlateCarree', L='LambertConformal', M='Mercator',
    N='NorthPolarStereo', G='Geostationary')
_scaleshort = dict(l='110m', i='50m', h='10m')


class PlotError(Exception):

    pass


class Plot:

    fontsize = dict(title=6, timestamp=5, mmnote=5, clabel=5, cbar=5,
        gridvalue=5, mmfilter=6, parameri=4, legend=6)
    linecolor = dict(coastline=_gray, country=_gray, province=_gray,
        city=_gray, county=_gray, parameri='k')
    linewidth = dict(coastline=0.3, country=0.3, province=0.2, city=0.1,
        county=0.1, parameri=0.3)

    def __init__(self, figsize=None, dpi=180, figure_boundary=None, inbox=False):
        self.mmnote = ''
        self.family = 'Lato'
        self.dpi = dpi
        if figsize is None:
            figsize = 7, 5
        self.fig = plt.figure(figsize=figsize)
        self.ax = None
        self.mpstep = 10
        self.mapset = None
        self.figure_boundary = figure_boundary
        self.inbox = inbox

    def setfamily(self, f):
        self.family = f

    def setfontsize(self, name, size):
        self.fontsize[name] = size

    def setlinecolor(self, name, color):
        self.linecolor[name] = color

    def setlinewidth(self, name, width):
        self.linewidth[name] = width

    def setmeriparastep(self, mpstep):
        self.mpstep = mpstep

    def setparameristep(self, mpstep):
        self.mpstep = mpstep

    def setdpi(self, d):
        self.dpi = d

    def setxy(self, georange, res):
        self.latmin, self.latmax, self.lonmin, self.lonmax = tuple(georange)
        self.x = np.arange(self.lonmin, self.lonmax+res, res)
        self.y = np.arange(self.latmin, self.latmax+res, res)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.res = res

    def setmap(self, key=None, proj=None, projection=None, aspect=None,
            resolution='i', **kwargs):
        """Set underlying map for the plot.

        Parameters
        ----------
        key : string, optional
            Shortcut key for built-in maps. Available options:
            chinaproper|chinamerc|chinalambert|euroasia|europe|
            northamerica|northpole (the default is None)
        proj : string, optional
            Cartopy-style projection names or shortcut names.
            Available shortcut options: P - PlateCarree|L -
            LambertConformal|M - Mercator|N - NorthPolarStereo|
            G - Geostationary (the default is None)
        projection : string, optional
            Basemap-style projection names. Only following options
            are allowed: cyl|merc|lcc|geos|npaepd (the default is
            None)
        aspect : string or float, optional
            Aspect ratio for lat/lon, only work in PlateCarree
            projection. If set, height of figure will be calculated
            by (lat_range / lon_range) * width_of_figure * aspect.
            This param is often used when representing data in mid-
            latitude regions in PlateCarree projection to offset
            projection distortion. (the default is None, which will
            fix aspect and change figure size. When set to 'auto',
            aspect will be calculated to fit the figure size.)
        resolution : string, optional
            Default scale of features (e.g. coastlines). Should be
            one of (l|i|h), which stands for 110m, 50m and 10m
            respectively. The default is 'i' (50m).

        """
        if 'resolution' in kwargs:
            kwargs.pop('resolution')
            warnings.warn('Param `resolution` is ignored in plotplus2.')
        if key is not None:
            proj, other_kwargs = self._from_map_key(key)
            kwargs.update(other_kwargs)
        if proj is None and projection is not None:
            _proj_dict = {'cyl':'P', 'merc':'M', 'lcc':'L', 'geos':'G', 'npaeqd':'N'}
            proj = _proj_dict.get(projection, None)
            if proj is None:
                raise PlotError('Only cyl/merc/lcc/geos/npaeqd are allowed in `projection` '
                                'param. If you want to use cartopy-style projection names, '
                                'please use `proj` param instead.')
        _projection = _projshort.get(proj.upper(), proj)
        if 'georange' in kwargs:
            georange = kwargs.pop('georange')
        if key == 'NorthPolarStereo' and 'boundinglat' in kwargs:
            georange = (kwargs.pop('boundinglat'), 90, -180, 180)
        self.proj = _projection
        self.trans = self.proj != 'PlateCarree'
        self.ax = plt.axes(projection=getattr(ccrs, _projection)(**kwargs))
        self.scale = _scaleshort[resolution]
        extent = georange[2:] + georange[:2]
        self.ax.set_extent(extent, crs=ccrs.PlateCarree())
        if aspect == 'auto':
            width, height = self.fig.get_size_inches()
            deltalon = georange[3] - georange[2]
            deltalat = georange[1] - georange[0]
            aspect_ratio = (height * deltalat) / (width * deltalon)
            self.ax.set_aspect(aspect_ratio)
        elif aspect is not None:
            self.ax.set_aspect(aspect)
        if self.figure_boundary is None:
            self.ax.outline_patch.set_linewidth(0)
        else:
            self.ax.outline_patch.set_linewidth(0.5)

    def _from_map_key(self, key):
        if key == 'chinaproper':
            proj = 'P'
            kwargs = {'georange':(20,40,100,130)}
        elif key == 'chinamerc':
            proj = 'M'
            kwargs = {'georange':(15,50,72.5,135)}
        elif key == 'chinalambert':
            proj = 'L'
            kwargs = {'georange':(15,55,80,125), 'central_longitude':102.5,
                'central_latitude':40, 'standard_parallels':(40,40)}
        elif key == 'euroasia':
            proj = 'L'
            kwargs = {'georange':(5,75,55,145), 'central_longitude':100,
                'central_latitude':40, 'standard_parallels':(40,40)}
        elif key == 'europe':
            proj = 'L'
            kwargs = {'georange':(5,75,-25,65), 'central_longitude':20,
                'central_latitude':40, 'standard_parallels':(40,40)}
        elif key == 'northamerica':
            proj = 'L'
            kwargs = {'georange':(5,75,-145,-55), 'central_longitude':-100,
                'central_latitude':40, 'standard_parallels':(40,40)}
        elif key == 'northpole':
            proj = 'N'
            kwargs = {'georange':(15,90,-180,180), 'central_longitude':105}
        return proj, kwargs

    def usemap(self, session):
        proj = session.mapproj.pop('proj')
        georange = session.mapproj.pop('georange')
        self.setmap(proj=proj, georange=georange, **session.mapproj)

    def _usemap(self, proj, georange=None, resolution='i'):
        self.ax = plt.axes(projection=proj)
        self.proj = type(proj).__name__
        self.trans = self.proj != 'PlateCarree'
        self.scale = _scaleshort[resolution]
        if georange:
            extent = georange[2:] + georange[:2]
            self.ax.set_extent(extent, crs=ccrs.PlateCarree())
        if self.figure_boundary is None:
            self.ax.outline_patch.set_linewidth(0)
        else:
            self.ax.outline_patch.set_linewidth(0.5)

    def usefeature(self, feature, facecolor=None, edgecolor=None, **kwargs):
        feature._kwargs.update(facecolor=facecolor, edgecolor=edgecolor)
        self.ax.add_feature(feature, **kwargs)

    def usemapset(self, mapset):
        self.mapset = mapset

    def drawcoastline(self, lw=None, color=None, res=None):
        lw = self.linewidth['coastline'] if lw is None else lw
        color = self.linecolor['coastline'] if color is None else color
        res = res if res else self.scale
        if self.mapset and self.mapset.coastline:
            self.usefeature(self.mapset.coastline, edgecolor=color, linewidth=lw)
        else:
            self.ax.add_feature(self.getfeature('physical', 'coastline', res,
                facecolor='none', edgecolor=color), linewidth=lw)

    def drawcountry(self, lw=None, color=None, res=None):
        lw = self.linewidth['country'] if lw is None else lw
        color = self.linecolor['country'] if color is None else color
        res = res if res else self.scale
        if self.mapset and self.mapset.country:
            self.usefeature(self.mapset.country, edgecolor=color, linewidth=lw)
        else:
            self.ax.add_feature(self.getfeature('cultural',
                'admin_0_boundary_lines_land', res, facecolor='none',
                edgecolor=color), linewidth=lw)

    @functools.lru_cache(maxsize=32)
    def getfeature(self, *args, **kwargs):
        return cfeature.NaturalEarthFeature(*args, **kwargs)

    def drawprovince(self, lw=None, color=None):
        lw = self.linewidth['province'] if lw is None else lw
        color = self.linecolor['province'] if color is None else color
        if self.mapset and self.mapset.province:
            self.usefeature(self.mapset.province, edgecolor=color, linewidth=lw)
        else:
            self.ax.add_feature(cfeature.ShapelyFeature(
                ciosr.Reader(_ProvinceDir).geometries(), ccrs.PlateCarree(),
                facecolor='none', edgecolor=color), linewidth=lw)

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
            color=color, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, self.mpstep))
        gl.ylocator = mticker.FixedLocator(np.arange(-80, 81, self.mpstep))
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

    def style(self, s):
        if s not in ('jma', 'bom'):
            print('Unknown style name. Only support jma or bom style.')
            return
        if s == 'jma':
            ocean_color = '#87A9D2'
            land_color = '#AAAAAA'
            self.linecolor.update(coastline='#666666', country='#666666',
                parameri='#666666',province='#888888', city='#888888')
        elif s == 'bom':
            ocean_color = '#E6E6FF'
            land_color = '#E8E1C4'
            self.linecolor.update(coastline='#D0A85E', country='#D0A85E',
                parameri='#D0A85E',province='#D0A85E', city='#D0A85E')
        if self.mapset and self.mapset.ocean:
            self.usefeature(self.mapset.ocean, color=ocean_color)
        else:
            self.ax.add_feature(cfeature.OCEAN.with_scale(self.scale),
                color=ocean_color)
        if self.mapset and self.mapset.land:
            self.usefeature(self.mapset.land, color=land_color)
        else:
            self.ax.add_feature(cfeature.LAND.with_scale(self.scale),
                color=land_color)

    def plot(self, *args, **kwargs):
        kwargs.update(transform=ccrs.PlateCarree())
        ret = self.ax.plot(*args, **kwargs)
        return ret

    def scatter(self, *args, **kwargs):
        kwargs.update(transform=ccrs.PlateCarree())
        ret = self.ax.scatter(*args, **kwargs)
        return ret

    def contour(self, data, clabel=True, clabeldict=dict(), ip=1, color='k', lw=0.5,
            vline=None, vlinedict=dict(), **kwargs):
        x, y, data = self.interpolation(data, ip)
        kwargs.update(colors=color, linewidths=lw, transform=ccrs.PlateCarree())
        c = self.ax.contour(x, y, data, **kwargs)
        if vline:
            vlinedict = merge_dict(vlinedict, {'color':color, 'lw':lw})
            if isinstance(vline, (int, float)):
                vline = [vline]
            for v in vline:
                if not isinstance(v, (int, float)):
                    raise ValueError('`{}` should be int or float'.format(v))
                try:
                    index = list(c.levels).index(v)
                except ValueError:
                    raise ValueError('{} not in contour levels'.format(v))
                else:
                    c.collections[index].set(**vlinedict)
        if clabel:
            if 'levels' in clabeldict:
                clabellevels = clabeldict.pop('levels')
            else:
                clabellevels = kwargs['levels']
            clabeldict.update(fmt='%d', fontsize=self.fontsize['clabel'])
            #labels = self.ax.clabel(c, clabellevels, **clabeldict)
            labels = self.ax.clabel(c, **clabeldict)
            for l in labels:
                l.set_family(self.family)
                if vline:
                    text = l.get_text()
                    for v in vline:
                        if str(v) == text:
                            l.set_color(vlinedict['color'])
        return c

    def contourf(self, data, gpfcmap=None, cbar=False, cbardict=dict(), ip=1,
            vline=None, vlinedict=dict(), **kwargs):
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
            if 'extend' in kwargs:
                cbardict.update(extend=kwargs.pop('extend'), extendfrac=0.02)
            self.colorbar(c, unit=kwargs.pop('unit', None), **cbardict)
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
        kwargs = merge_dict(kwargs, dict(size='2%', pad='1%'))
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
            axes_class=plt.Axes)
        cb = self.fig.colorbar(mappable, orientation=orientation, cax=cax, **kwargs)
        self.fig.sca(self.ax)
        cb.ax.tick_params(labelsize=self.fontsize['cbar'], length=1.5)
        cb.outline.set_linewidth(0.3)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family(self.family)
        return cb

    def sidebar(self, mappable, unit=None, **kwargs):
        ticks = kwargs.pop('ticks')
        ticks = [ticks[0], ticks[-1]]
        del kwargs['size'], kwargs['pad']
        cax = self.fig.add_axes([0.18, 0.13, 0.01, 0.05])
        cb = self.fig.colorbar(mappable, cax=cax, ticks=ticks, **kwargs)
        cb.ax.tick_params(labelsize=self.fontsize['cbar'], length=0)
        cb.outline.set_linewidth(0.1)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family(self.family)
        plt.sca(self.ax)
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
        vs = self.stepcal(num)
        q = self.ax.quiver(self.x[::vs], self.y[::vs], u[::vs, ::vs], v[::vs, ::vs], **kwargs)
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

    def pcolormesh(self, data, gpfcmap=None, cbar=False, cbardict=dict(), ip=1, **kwargs):
        if gpfcmap:
            import matplotlib.colors as mclr
            gpfdict = gpf.cmap(gpfcmap)
            cmap = gpfdict.pop('cmap')
            levels = gpfdict.pop('levels')
            norm = mclr.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            kwargs.update(cmap=cmap, norm=norm)
        x, y, data = self.interpolation(data, ip)
        ret = self.ax.pcolormesh(x, y, data, **kwargs)
        if cbar:
            if 'ticks' not in cbardict:
                levels = gpfdict['levels']
                step = len(levels) // 40 + 1
                cbardict.update(ticks=levels[::step])
            cbardict.update(size='2%', pad='1%')
            if 'extend' in gpfdict:
                cbardict.update(extend=gpfdict.pop('extend'), extendfrac=0.02)
            self.colorbar(ret, unit=gpfdict.pop('unit', None), **cbardict)
        return ret

    def _get_stroke_patheffects(self):
        import matplotlib.patheffects as mpatheffects
        return [mpatheffects.Stroke(linewidth=1, foreground='w'), mpatheffects.Normal()]

    def gridvalue(self, data, num=20, fmt='{:d}', color='b', fontsize=None,
            stroke=False, **kwargs):
        if fontsize is None:
            fontsize = self.fontsize['gridvalue']
        if stroke:
            kwargs.update(path_effects=self._get_stroke_patheffects())
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

    def marktext(self, x, y, text='', mark='×', textpos='right', stroke=False,
            bbox=dict(), family='plotplus', markfontsize=None, **kwargs):
        if family == 'plotplus':
            kwargs.update(family=self.family)
        elif family is not None:
            kwargs.update(family=family)
        if not markfontsize:
            markfontsize = self.fontsize['mmfilter']
        fontsize = kwargs.pop('fontsize', self.fontsize['marktext'])
        if stroke:
            kwargs.update(path_effects=self._get_stroke_patheffects())
        kwargs.update(transform=ccrs.PlateCarree())
        bbox = merge_dict(bbox, {'facecolor':'none', 'edgecolor':'none'})
        xy, xytext, ha, va=dict(right=((1, 0.5), (2, 0), 'left', 'center'),
                                left=((0, 0.5), (-2, 0), 'right', 'center'),
                                top=((0.5, 1), (0, 1), 'center', 'bottom'),
                                bottom=((0.5, 0), (0, -1), 'center', 'top')).get(textpos)
        an_mark = self.ax.annotate(mark, xy=(x,y), xycoords='data', va='center',
              ha='center', bbox=bbox, fontsize=markfontsize, **kwargs)
        an_text = self.ax.annotate(text, xy=xy, xycoords=an_mark, xytext=xytext,
              textcoords='offset points', va=va, ha=ha, bbox=bbox,
              fontsize=fontsize, **kwargs)
        return an_text

    def maxminfilter(self, data, type='min', fmt='{:d}', weight='bold', color='b',
            fontsize=None, window=15, vmin=-1e7, vmax=1e7, stroke=False, marktext=False,
            marktextdict=dict(), **kwargs):
        '''Use res keyword or ip keyword to interpolate'''
        if fontsize is None:
            fontsize = self.fontsize['mmfilter']
        if stroke:
            kwargs.update(path_effects=self._get_stroke_patheffects())
        textfunc = self.ax.text
        kwargs.update(fontweight=weight, color=color, fontsize=fontsize, ha='center',
            va='center', transform=ccrs.PlateCarree())
        if marktext:
            argsdict = dict(fontsize=fontsize, weight=weight, color=color, stroke=stroke,
                markfontsize=8, family=None, textpos='bottom')
            kwargs = merge_dict(marktextdict, argsdict)
            textfunc = self.marktext
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
                textfunc(x*res+self.lonmin, y*res+self.latmin, fmt.format(d), **kwargs)

    def title(self, s, nasdaq=False):
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
                fromhour, tohour, fromtime.strftime(stdfmt), totime.strftime(stdfmt))
        self._timestamp(s)

    def _timestamp(self, s):
        self.ax.text(0, 1.01, s, transform=self.ax.transAxes,
            fontsize=self.fontsize['timestamp'], family=self.family)

    def _colorbar_unit(self, s):
        if not s:
            return
        self.ax.text(1.05, 1.01, s, transform=self.ax.transAxes, ha='right',
            fontsize=self.fontsize['timestamp'], family=self.family)

    def maxminnote(self, data, name, unit, type='max', fmt='{:.1f}'):
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
        notestr = '{:s} {:s}: '.format(typestr, name) + fmt.format(notevalue) + ' ' + unit
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


class MapSet:

    def __init__(self, coastline=None, country=None, land=None, ocean=None,
            province=None, city=None, county=None):
        self.coastline = coastline
        self.country = country
        self.land = land
        self.ocean = ocean
        self.province = province
        self.city = city
        self.county = county

    @classmethod
    def from_natural_earth(cls, scale, extent, country=True, land=False, ocean=False):
        ins = cls()
        ins.coastline = PartialNaturalEarthFeature('physical', 'coastline',
            scale, extent=extent)
        if country:
            ins.country = PartialNaturalEarthFeature('cultural',
                'admin_0_boundary_lines_land', scale, extent=extent)
        if land:
            ins.land = PartialNaturalEarthFeature('physical', 'land', scale,
                extent=extent)
        if ocean:
            ins.ocean = PartialNaturalEarthFeature('physical', 'ocean',
                scale, extent=extent)
        return ins

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class PartialShapelyFeature(cfeature.ShapelyFeature):

    def __init__(self, *args, extent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extent = extent
        self.make_partial()

    def intersecting_geometries(self, extent):
        return self.geometries()

    def make_partial(self):
        self._geoms = super().intersecting_geometries(self.extent)


class PartialNaturalEarthFeature(cfeature.NaturalEarthFeature):

    def __init__(self, *args, extent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extent = extent
        self._geoms = ()
        self.make_partial()

    def intersecting_geometries(self, extent):
        return self.geometries()

    def geometries(self):
        return iter(self._geoms)

    def make_partial(self):
        path = ciosr.natural_earth(resolution=self.scale,
            category=self.category, name=self.name)
        self._geoms = tuple(ciosr.Reader(path).geometries())
        self._geoms = super().intersecting_geometries(self.extent)

