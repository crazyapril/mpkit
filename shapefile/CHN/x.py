import shapefile

def main():
    sf = shapefile.Reader('CHN_adm2.shp')
    prov = dict()
    county = dict()
    #{1:{'Anhui', [('安庆', 'Anqing'), (...)]}}
    for i , sr in enumerate(sf.shapeRecords()):
        r = sr.record
        province_num = r[3]
        province_name = r[4]
        if province_num not in prov:
            prov[province_num] = dict(name=province_name, city=dict())
            county[province_num] = dict()
        city_name = r[-2]
        city_name_eng = r[6]
        if '|' in city_name:
            city_name = city_name.split('|')[-1]
        prov[province_num]['city'][city_name_eng] = city_name
    sf2 = shapefile.Reader('CHN_adm3.shp')
    for i, sr in enumerate(sf2.shapeRecords()):
        r = sr.record
        province_num = r[3]
        city_name_eng = r[6]
        county_name_eng = r[8]
        county_name = r[-2]
        if len(county_name) > 60:
            replace = input('**%s %s %s>' % (prov[province_num]['name'], prov[province_num]['city'][city_name_eng], county_name_eng))
            if replace == '':
                county_name = county_name_eng
            else:
                county_name = replace
        if city_name_eng not in county[province_num]:
            county[province_num][city_name_eng] = list()
        county[province_num][city_name_eng].append((county_name, county_name_eng))
    f = open('data2.txt', 'w')
    for p in prov:
        f.write('%d %s\n' % (p, prov[p]['name']))
        for c in prov[p]['city']:
            try:
                f.write('>%s %s\n' % (prov[p]['city'][c], c))
            except:
                f.write('>Something Error.\n')
            if c in county[p]:
                for t in county[p][c]:
                    try:
                        f.write('>>%s %s\n' % (t[0], t[1]))
                    except:
                        f.write('>>Something Error.\n')
            f.write('\n')
        f.write('\n')
    f.close()

main()
