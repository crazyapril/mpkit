import mpkit.easygrib as easygrib
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap

_accepted_mapkey_ = ['asia', 'chinaproper', 'europe', 'northamerica', 'northpole']

class ModelGo:

    def __init__(self):
        self.Flags = dict(fixed_georange=False, accum=False, two_time=False,
                          default_zero=False, fixed_map=False, choose_map=False)
        self.Settings = dict(accum=list(), interval=6)
        self.Params = list()
        self.Data = dict()
        #easygrib.installproxy()

    def choose_map(self):
        choice = 0
        for i, mapkey in enumerate(_accepted_mapkey_, 1):
            print('{:d}. {:s} '.format(i, mapkey), end='')
        print()
        while choice < 1 or choice > len(_accepted_mapkey_):
            choice = int(input('输入需要的地图>'))
        self.set_fixed_map(_accepted_mapkey_[choice-1])

    def set_choose_map(self):
        self.Flags['choose_map'] = True

    def set_step(self, step, dateoffset=0):
        self.Settings['step'] = step
        self.Settings['dateoffset'] = dateoffset

    def set_fixed_georange(self, georange):
        self.Flags['fixed_georange'] = True
        self.georange = georange

    def set_fixed_map(self, mapkey):
        mapkey = mapkey.lower()
        if mapkey not in _accepted_mapkey_:
            raise KeyError('Mapkey {:s} is not supported.'.format(mapkey))
        self.Flags['fixed_map'] = True
        if mapkey == 'asia':
            self.set_fixed_georange((-5, 80, 0, 200))
            self.map = Basemap(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=60, urcrnrlon=200,
                          lat_0=42.5, lon_0=100)
        elif mapkey == 'chinaproper':
            self.set_fixed_georange((20, 40, 100, 130))
            self.map = Basemap(projection='cyl', resolution='i',
                          llcrnrlat=20, urcrnrlat=40, llcrnrlon=100, urcrnrlon=130)
        elif mapkey == 'europe':
            self.set_fixed_georange((-15, 80, 0, 360))
            self.map = Basemap(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=0, urcrnrlon=140,
                          lat_0=42.5, lon_0=40)
        elif mapkey == 'northamerica':
            self.set_fixed_georange((-15, 80, 150, 360))
            self.map = Basemap(projection='lcc', resolution='l',
                          llcrnrlat=-5, urcrnrlat=45, llcrnrlon=220, urcrnrlon=360,
                          lat_0=42.5, lon_0=260)
        elif mapkey == 'northpole':
            self.set_fixed_georange((-15, 90, 0, 360))
            self.map = Basemap(projection='npaeqd', resolution='l',
                          boundinglat=15, lon_0=105, round=False)

    def set_model(self, model, *args):
        self.model = model
        self.Params.extend(list(args))

    def set_accum(self, *args):
        self.Flags['accum'] = True
        self.Settings['accum'].extend(list(args))

    def set_two_time(self):
        self.Flags['two_time'] = True

    def set_default_zero(self):
        self.Flags['default_zero'] = True

    def set_plot(self, f):
        self.plot_func = f

    def getdata(self, time=False):
        for p in self.Params:
            if self.Flags['accum'] and not time and p in self.Settings['accum']:
                self.Data[p, 'prev'] = self.Data.get(p, 0)
            key = (p, self.fcsthour) if time else p
            try:
                self.Data[key] = easygrib.ModelData(self.model, self.basetime, self.fcsthour,
                                                  self.georange, **self.model_dict(p))
            except ValueError:
                if self.Flags['default_zero']:
                    self.Data[key] = 0
                else:
                    raise
                
    def model_dict(self, p):
        if self.model == 'GFS':
            return dict(level=p[0], parameter=p[1])
        elif self.model == 'EC':
            if len(p) > 1:
                return dict(product=p[0], mode=p[1])
            else:
                return dict(product=p)
        elif self.model == 'ERA':
            return dict(level=p[0], parameter=p[1])
        
    def run(self):
        basetime = input('输入模式时间>')
        key = input('输入预报时间>')
        if self.Flags['choose_map']:
            self.choose_map()
        if not self.Flags['fixed_georange']:
            self.georange = eval(input('输入数据范围>'))
        BTG = BaseTimeGen(basetime)
        for bt in BTG.time:
            self.basetime = bt
            TG = TimeGen(self.basetime, self.Settings['step'], self.Settings['dateoffset'])
            TG.input(key)
            if self.Flags['two_time']:
                for fcsthour in TG.time:
                    self.fcsthour = fcsthour
                    self.getdata(time=True)
                self.fcsthour = TG.time[0]
                self.fcsthour2 = TG.time[1]
                self.plot_func(self)
            else:
                for fcsthour in TG.time:
                    print(fcsthour)
                    self.fcsthour = fcsthour
                    self.getdata()
                    self.plot_func(self)

class TimeGen:

    def __init__(self, basetime, step=24, dateoffset=0):
        if isinstance(basetime, datetime):
            self.basetime = basetime
        else:
            self.basetime = datetime.strptime(basetime, '%Y%m%d%H')
        self.step = step
        self.dateoffset = dateoffset

    def input(self, key):
        self.time = list()
        self.gen(key)

    def gen(self, key):
        if ' ' in key:
            for k in key.split():
                self.gen(k)
            return
        elif '~' in key:
            return self.stepgen(key)
        else:
            return self.singlegen(key)

    def stepgen(self, key):
        sharp = False
        elements = tuple(key.split('~'))
        if len(elements) == 2:
            start, stop = elements
            step = None
        elif len(elements) == 3:
            start, stop, step = elements
        if start.startswith('d'):
            if step:
                step = int(step) * 24
            else:
                step = 24
        else:
            if not step:
                step = self.step
            else:
                if step.endswith('!'):
                    sharp = True
                    step = step[:-1]
                step = int(step)
        starttime = self.convert(start)
        stoptime = self.convert(stop)
        if sharp:
            start_n = (starttime // step + 1) * step
            stop_n = stoptime // step * step
            locallist = list(range(start_n, stop_n+1, step))
            if starttime % step != 0:
                locallist.insert(0, starttime)
            if stoptime % step != 0:
                locallist.append(stoptime)
            self.time.extend(locallist)
        else:
            self.time.extend(list(range(starttime, stoptime+1, step)))

    def singlegen(self, key):
        self.time.append(self.convert(key))

    def convert(self, key):
        if len(key) == 10:
            t = datetime.strptime(key, '%Y%m%d%H')
            return (t - self.basetime) // timedelta(hours=1)
        elif len(key) == 8:
            t = datetime.strptime(key + '00', '%Y%m%d%H')
            return (t - self.basetime) // timedelta(hours=1) + self.dateoffset
        else:
            return int(key)

class BaseTimeGen:

    def __init__(self, inputstr):
        self.time = list()
        basetime = inputstr[:10]
        TG = TimeGen(basetime, 6)
        TG.input(inputstr)
        for t in TG.time:
            self.time.append((TG.basetime + timedelta(hours=t)).strftime('%Y%m%d%H'))
