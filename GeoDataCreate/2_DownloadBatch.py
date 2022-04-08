import json
import ee

# Trigger the authentication flow
ee.Authenticate()

# Initialise the library
ee.Initialize()

# Load geojson of region of interest
cement_china = 'users/gvsametrino/China_2'

china_landcovers = 'users/gvsametrino/LC_Cement_1kmpolys'


def unpack(thelist):
    unpacked = []
    for i in thelist:
        unpacked.append(i[0])
        unpacked.append(i[1])
    return unpacked


def ee_img_chips(month, year, shp):
    '''
    expected input parameters :

        month : 1 or 4
        year : 2018 or 2019

        shp : link to GEE asset
        cement_china_shp OR china_landcovers
    '''

    fc = ee.FeatureCollection(shp)

    if month == 1 and year == 2018:
        src = ee.Image('LANDSAT/LC08/C01/T1_32DAY_TOA/20180101')
        # Single band
        # b10 = src.select(['B10'])
        # b6 = src.select(['B6'])
        # band combination
        # b76 = (src.select(['B7'])).divide(src.select(['B6']))
        # Normalized Difference Vegetation Index (NDVI)
        # b43 = (src.select(['B4']).subtract(src.select(['B3']))).divide((src.select(['B4'])).add(src.select(['B3'])))
        # Colour RGB images
        b234 = (src.select(['B4']).add(src.select(['B3'])).add(src.select(['B2'])))
    elif month == 4 and year == 2018:
        src = ee.Image('LANDSAT/LC08/C01/T1_32DAY_TOA/20180407')
        # b10 = src.select(['B10'])
        # b6 = src.select(['B6'])
        # b76 = (src.select(['B7'])).divide(src.select(['B6']))
        b234 = (src.select(['B4']).add(src.select(['B3'])).add(src.select(['B2'])))
    elif month == 1 and year == 2017:
        src = ee.Image('LANDSAT/LC08/C01/T1_32DAY_TOA/20170101')
        # b10 = src.select(['B10'])
        # b6 = src.select(['B6'])
        # b76 = (src.select(['B7'])).divide(src.select(['B6']))
        b234 = (src.select(['B4']).add(src.select(['B3'])).add(src.select(['B2'])))
    elif month == 4 and year == 2017:
        src = ee.Image('LANDSAT/LC08/C01/T1_32DAY_TOA/20170407')
        # b10 = src.select(['B10'])
        # b6 = src.select(['B6'])
        # b76 = (src.select(['B7'])).divide(src.select(['B6']))
        b234 = (src.select(['B4']).add(src.select(['B3'])).add(src.select(['B2'])))
    else:
        print(f'please add image src for date: {month, year}')

    # adding this helps clip image by each feature in shp
    img = b234.clipToCollection(fc)
    # img = b10.clipToCollection(fc)
    featlist = fc.getInfo()["features"]

    # landcover or cement site:
    if shp == cement_china:
        num = 'uid'
        # imgstart = "b10"
        imgstart = "b234"
    elif shp == china_landcovers:
        num = 'ID'
        # imgstart = "Landcover_b10_"
        imgstart = "Landcover_b234_"
    else:
        print('please enter lc or id')

    # -*- coding: utf-8 -*-
    """
    @author: Maral Bayaraa, May 2020

    Edited and adapted from https://gis.stackexchange.com/questions/237775/google-earth-engine-clip-image-into-multiple-regions-defined-by-a-shapefile-and/240834
    @author-original: Rodrigo E. Principe

    email: fitoprincipe82 at gmail
    github: https://github.com/gee-community/gee_tools
    twitter: @fitoprincipe
    """

    for f in featlist[100:105]:
        # [100:105]:

        geomlist = unpack(f["geometry"]["coordinates"][0])
        geom = ee.Geometry.Polygon(geomlist)

        feat = ee.Feature(geom)

        # get the metadata
        disS = f["properties"][num]

        imgname = str(imgstart) + str(year) + '0' + str(month)

        try:
            task = ee.batch.Export.image.toDrive(
                image=img,
                description=imgname + '_' + str(disS),
                folder=imgname,
                fileNamePrefix=imgname + '_' + str(disS),
                region=feat.geometry().bounds().getInfo()["coordinates"],
                scale=30)

            task.start()
            print("exporting {0} {1}..".format(imgname, disS))

        except:
            print("failed img : {0} {1}..".format(imgname, disS))
            continue


# Call the function for January 2017 data for cement plants
ee_img_chips(1, 2017, cement_china)

# Call the function for April 2017 data for cement plants
ee_img_chips(4, 2017, cement_china)

# Call the function for January 2018 data for cement plants
ee_img_chips(1, 2018, cement_china)

# Call the function for January 2017 data for land cover around cement plants
ee_img_chips(1, 2017, china_landcovers)

# Call the function for April 2017 data for land cover around cement plants
ee_img_chips(4, 2017, china_landcovers)

# Call the function for January 2018 data for land cover around cement plants
ee_img_chips(1, 2018, china_landcovers)

# Call the function for April 2018 data for land cover around cement plants
ee_img_chips(4, 2018, china_landcovers)
