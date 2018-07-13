import mpkit.gpf as gpf
import os, sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap, interp, shiftgrid
from scipy.ndimage.filters import maximum_filter, minimum_filter

#Plotplus 0.1 - 2016/2/16
#Plotplus 0.2
_ShapeFileDirectory = os.path.join(os.path.split(__file__)[0], 'shapefile')
_ProvinceDirectory = os.path.join(_ShapeFileDirectory, 'CP/ChinaProvince')
_CityDirectory = os.path.join(_ShapeFileDirectory, 'CHN/CHN_adm2')
_CitySDirectory = os.path.join(_ShapeFileDirectory, 'TWN/TWN_adm2')
_CountyDirectory = os.path.join(_ShapeFileDirectory, 'CHN/CHN_adm3')

class PlotError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class Plot:

    def __init__(self, figsize=(7,5), dpi=180):
        self.mmnote = ''
        self.family = 'Lato'
        self.dpi = dpi
        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.gca()
        self.fontsize = dict(title=6, timestamp=5, mmnote=5, clabel=5, cbar=5,
                             gridvalue=5, mmfilter=6, parameri=4, legend=6,
                             marktext=5)
        self.linecolor = dict(coastline='#222222', country='#222222', province='#222222',
                              city='#222222', county='#222222', parameri='k')
        self.mpstep = 10

    def setfamily(self, f):
        self.family = f

    def setfontsize(self, name, size):
        self.fontsize.update({name:size})

    def setlinecolor(self, name, size):
        self.linecolor.update({name:size})

    def setdpi(self, dpi):
        self.dpi = dpi

    def setmeriparastep(self, mpstep):
        self.mpstep = mpstep

    def setxy(self, georange, res):
        self.latmin, self.latmax, self.lonmin, self.lonmax = tuple(georange)
        if self.lonmin == 0 and self.lonmax == 360:
            self.x = np.arange(0, 360, res)
        else:
            self.x = np.arange(self.lonmin, self.lonmax+res, res)
        self.y = np.arange(self.latmin, self.latmax+res, res)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.res = res
        if self.trans:
            self.nx, self.ny = self.x, self.y
            self.x, self.y = self.m(self.xx, self.yy)

    def setmap(self, key=None, projection='cyl', resolution='i', **kwargs):
        if key == 'chinaproper':
            kwargs.update(projection='cyl', resolution='i',
                          llcrnrlat=20, urcrnrlat=40, llcrnrlon=100, urcrnrlon=130)
        elif key == 'chinamerc':
            kwargs.update(projection='merc', resolution='i',
                          llcrnrlat=15, urcrnrlat=55, llcrnrlon=70, urcrnrlon=140,
                          lat_0=15, lon_0=95)
        elif key == 'euroasia':
            kwargs.update(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=60, urcrnrlon=200,
                          lat_0=42.5, lon_0=100)
        elif key == 'europe':
            kwargs.update(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=0, urcrnrlon=140,
                          lat_0=42.5, lon_0=40)
        elif key == 'northamerica':
            kwargs.update(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=220, urcrnrlon=360,
                          lat_0=42.5, lon_0=260)
        elif key == 'northpolar':
            kwargs.update(projection='npaeqd', resolution='l',
                          boundinglat=15, lon_0=105, round=False)
        if 'georange' in kwargs:
            georange = kwargs.pop('georange')
            kwargs.update(llcrnrlat=georange[0], urcrnrlat=georange[1],
                          llcrnrlon=georange[2], urcrnrlon=georange[3])
        kwargs.update(ax=self.ax, projection=projection, resolution=resolution)
        if projection == 'cyl':
            if 'fix_aspect' not in kwargs:
                kwargs.update(fix_aspect=True)
            self.trans = False
        else:
            self.trans = True
        self.m = Basemap(**kwargs)
        self.ax = plt.gca()

    def usemap(self, m):
        self.m = m
        if self.m.projection != 'cyl':
            self.trans = True
        else:
            self.trans = False

    def style(self, s):
        if s == 'jma':
            self.m.drawmapboundary(fill_color='#87A9D2', ax=self.ax)
            self.m.fillcontinents(color = '#AAAAAA',lake_color='#87A9D2', ax=self.ax)
            self.linecolor.update(dict(coastline='#666666', country='#666666',
                                       parameri='#666666',province='#888888', city='#888888'))
        elif s == 'bom':
            self.m.drawmapboundary(fill_color='#E6E6FF')
            self.m.fillcontinents(color='#E8E1C4', lake_color='#E6E6FF')
            self.linecolor.update(dict(coastline='#D0A85E', country='#D0A85E',
                                       parameri='#D0A85E',province='#D0A85E', city='#D0A85E'))

    def drawcoastline(self, lw=0.3, color=None):
        if color is None:
            color = self.linecolor['coastline']
        self.m.drawcoastlines(linewidth=lw, color=color, ax=self.ax)

    def drawcountries(self, lw=0.3, color=None):
        if color is None:
            color = self.linecolor['country']
        self.m.drawcountries(linewidth=lw, color=color, ax=self.ax)

    def drawprovinces(self, lw=0.2, color=None):
        if color is None:
            color = self.linecolor['province']
        self.m.readshapefile(_ProvinceDirectory, 'Province', linewidth=lw,
                             color=color, ax=self.ax)

    def drawcities(self, lw=0.1, color=None):
        if color is None:
            color = self.linecolor['city']
        self.m.readshapefile(_CityDirectory, 'City', linewidth=lw, color=color,
                             ax=self.ax)
        self.m.readshapefile(_CitySDirectory, 'CityS', linewidth=lw, color=color,
                             ax=self.ax)

    def drawcounties(self, lw=0.1, color=None):
        if color is None:
            color = self.linecolor['county']
        self.m.readshapefile(_CountyDirectory, 'County', linewidth=lw, color=color, ax=self.ax)

    def drawparameri(self, line=None, color=None, lw=0.3, fontsize=None, **kwargs):
        if line is None and not self.trans:
            lw = 0
        if fontsize is None:
            fontsize = self.fontsize['parameri']
        if color is None:
            color = self.linecolor['parameri']
        kwargs.update(linewidth=lw, fontsize=fontsize, color=color, family=self.family)
        self.m.drawparallels(np.arange(-80,80,self.mpstep), labels=[1,0,0,0], ax=self.ax, **kwargs)
        self.m.drawmeridians(np.arange(0,360,self.mpstep), labels=[0,0,0,1], ax=self.ax, **kwargs)

    def draw(self, cmd):
        if ' ' in cmd:
            for c in cmd.split():
                self.draw(c)
        else:
            if cmd == 'coastline' or cmd == 'coastlines':
                self.drawcoastline()
            elif cmd == 'country' or cmd == 'countries':
                self.drawcountries()
            elif cmd == 'province' or cmd == 'provinces':
                self.drawprovinces()
            elif cmd == 'city' or cmd == 'cities':
                self.drawcities()
            elif cmd == 'county' or cmd == 'counties':
                self.drawcounties()
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
            if self.trans:
                nx, ny = self.m(newx, newy)
            ndata = interp(data, self.x, self.y, newx, newy, order=3)
            return nx, ny, ndata

    def stepcal(self, num, ip=1):
        totalpt = (self.lonmax - self.lonmin) / self.res * ip
        return int(totalpt / num)

    def legend(self, lw=0., **kwargs):
        rc = dict(loc='upper right', framealpha=0.)
        rc.update(kwargs)
        ret = self.ax.legend(prop=dict(size=self.fontsize['legend'], family=self.family),
                             **rc)
        ret.get_frame().set_linewidth(lw)
        return ret

    def plot(self, *args, **kwargs):
        ret = self.ax.plot(*args, **kwargs)
        return ret

    def scatter(self, *args, **kwargs):
        ret = self.ax.scatter(*args, **kwargs)
        return ret

    def contour(self, data, clabel=True, clabeldict=dict(), ip=1, color='k', lw=0.5,
                filter=None, vline=None, vlinedict=dict(), **kwargs):
        x, y, data = self.interpolation(data, ip)
        kwargs.update(colors=color, linewidths=lw)
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
        if filter is not None:
            self.maxminfilter(data, res=self.res/ip, **filter)
        return c

    def contourf(self, data, gpfcmap=None, cbar=False, cbardict=dict(), ip=1,
                 vline=None, vlinedict=dict(), **kwargs):
        if gpfcmap:
            kwargs = merge_dict(kwargs, gpf.cmap(gpfcmap))
        x, y, data = self.interpolation(data, ip)
        c = self.ax.contourf(x, y, data, **kwargs)
        if cbar:
            if 'ticks' not in cbardict:
                levels = kwargs['levels']
                step = len(levels) // 40 + 1
                cbardict.update(ticks=levels[::step])
            cbardict.update(size='2%', pad='1%')
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
        if kwargs.pop('sidebar', False):
            return self.sidebar(mappable, unit, **kwargs)
        if kwargs.pop('orientation', None) == 'horizontal':
            cb = self.m.colorbar(mappable, 'bottom', **kwargs)
        else:
            cb = self.m.colorbar(mappable, **kwargs)
            self._colorbar_unit(unit)
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
        kwargs.update(color=color, linewidth=lw, density=density)
        ret = self.ax.streamplot(self.x, self.y, u, v, **kwargs)
        return ret

    def barbs(self, u, v, color='k', lw=0.5, length=4, num=12, **kwargs):
        kwargs.update(color=color, linewidth=lw, length=length)
        bs = self.stepcal(num)
        xbs, ybs, ubs, vbs = self.x[::bs], self.y[::bs], u[::bs,::bs], v[::bs,::bs]
        nh = np.meshgrid(xbs, ybs)[1] >= 0
        unh = np.ma.masked_where(~nh, ubs)
        vnh = np.ma.masked_where(~nh, vbs)
        ret = self.ax.barbs(xbs, ybs, unh, vnh, **kwargs)
        ush = np.ma.masked_where(nh, ubs)
        vsh = np.ma.masked_where(nh, vbs)
        retsh = self.ax.barbs(xbs, ybs, ush, vsh, flip_barb=True, **kwargs)
        return ret, retsh

    def quiver(self, u, v, num=40, scale=None, qkey=False, qkeydict=dict(), **kwargs):
        kwargs.update(width=0.0015, headwidth=3)
        if self.trans:
            if scale is None:
                scale = 500
            kwargs.update(scale=scale)
            uu, nx = shiftgrid(180., u, self.nx, start=False)
            vv, nx = shiftgrid(180., v, self.nx, start=False)
            uproj, vproj, xx, yy = self.m.transform_vector(uu, vv, nx, self.ny, num, num,
                                                           returnxy=True)
            q = self.m.quiver(xx, yy, uproj, vproj, **kwargs)
        else:
            if scale is None:
                scale = 500
            kwargs.update(scale=scale)
            vs = self.stepcal(num)
            q = self.m.quiver(self.x[::vs], self.y[::vs], u[::vs,::vs], v[::vs,::vs], **kwargs)
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

    def gridvalue(self, data, num=20, fmt='%d', color='b', fontsize=None,
                  stroke=False, **kwargs):
        if fontsize is None:
            fontsize = self.fontsize['gridvalue']
        if stroke:
            kwargs.update(path_effects=self._get_stroke_patheffects())
            #import matplotlib.patheffects as mpatheffects
            #kwargs.update(path_effects=[mpatheffects.withSimplePatchShadow(
            #    offset=(0.5,-0.5), alpha=0.7)])
        step = self.stepcal(num)
        kwargs.update(color=color, fontsize=fontsize, ha='center', va='center',
                      family=self.family)
        meri, para = len(self.y), len(self.x)
        for i in range(1, meri-1, step):
            for j in range(1, para-1, step):
                self.ax.text(j*self.res+self.lonmin, i*self.res+self.latmin,
                             fmt % (data[i][j]), **kwargs)

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

    def maxminfilter(self, data, type='min', fmt='%d', weight='bold', color='b',
                     fontsize=None, window=15, vmin=-1e7, vmax=1e7, stroke=False,
                     marktext=False, marktextdict=dict(), **kwargs):
        '''Use res keyword or ip keyword to interpolate'''
        if fontsize is None:
            fontsize = self.fontsize['mmfilter']
        if stroke:
            kwargs.update(path_effects=self._get_stroke_patheffects())
        textfunc = self.ax.text
        kwargs.update(fontweight=weight, color=color, fontsize=fontsize,
                      ha='center', va='center')
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
            ftr = minimum_filter
        elif type == 'max':
            ftr = maximum_filter
        else:
            raise PlotError('Unsupported filter type!')
        dataftr = ftr(data, window, mode='reflect')
        yind, xind = np.where(data == dataftr)
        ymax, xmax = data.shape
        for y, x in zip(yind, xind):
            d = data[y, x]
            if d < vmax and d > vmin and x not in (0, xmax-1) and y not in (0, ymax-1):
                textfunc(x*res+self.lonmin, y*res+self.latmin, fmt % d, **kwargs)
    
    def _get_stroke_patheffects(self):
        import matplotlib.patheffects as mpatheffects
        return [mpatheffects.Stroke(linewidth=1, foreground='w'), mpatheffects.Normal()]

    def title(self, s, nasdaq=True):
        if nasdaq:
            s = s + ' @NASDAQ'
        self.ax.text(0, 1.04, s, transform=self.ax.transAxes, fontsize=self.fontsize['title'],
                     family=self.family)

    def timestamp(self, basetime, fcsthour, duration=0, nearest=None):
        stdfmt = '%Y/%m/%d %a %HZ'
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
