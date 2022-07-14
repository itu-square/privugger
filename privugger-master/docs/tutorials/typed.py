import aesara
import aesara.tensor as at
import numpy as np


@aesara.compile.ops.as_op(itypes=[at.lvector, at.lvector, at.lvector],
    otypes=[at.lmatrix])
def method(countries, zips, ages):

    def all_attributes_sorted(countries, zips, ages):
        output = []
        rd_countries = np.repeat(0, countries.size)
        rd_zips = np.repeat(0, zips.size)
        rd_ages = np.repeat(0, ages.size)
        for i in range(100):
            if 50 < countries[i] <= 100:
                rd_countries[i] = 2
            elif countries[i] <= 50:
                rd_countries[i] = 1
            elif 100 < countries[i] <= 150:
                rd_countries[i] = 3
            elif 150 < countries[i] <= 195:
                rd_countries[i] = 4
            if 25 < zips[i] <= 50:
                rd_zips[i] = 2
            elif zips[i] <= 25:
                rd_zips[i] = 1
            elif 50 < zips[i] <= 75:
                rd_zips[i] = 3
            elif 75 < zips[i] <= 100:
                rd_zips[i] = 4
            if 1977 < ages[i] <= 1992:
                rd_ages[i] = 2
            elif ages[i] <= 1977:
                rd_ages[i] = 1
            elif 1992 < ages[i] <= 2003:
                rd_ages[i] = 3
        output.append(rd_countries)
        output.append(rd_zips)
        output.append(rd_ages)
        return np.array(output)
    return all_attributes_sorted(countries, zips, ages)
