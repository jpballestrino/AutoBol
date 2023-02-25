"""
Copyright (C) 2021  Juan Pedro Ballestrino, Cecilia Deandraya & Cristian Uviedo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from PIL import Image

def save_predict(data,salida,verbose):

    if verbose:
        if not os.path.exists(salida + '/Clasificacion.csv'):  # create output di
            data.to_csv(salida + '/Clasificacion.csv', index=False)
        else:
            data.to_csv(salida + '/Clasificacion.csv', mode='a', index=False, header=False)
    else:
        if not os.path.exists(salida + '/Predicciones.csv'):  # create output di
            data.to_csv(salida + '/Predicciones.csv', index=False)
        else:
            data.to_csv(salida + '/Predicciones.csv', mode='a', index=False, header=False)
    pass


def save_img(img,fname,i,fout):
    img.save(fout+'/'+fname[i:i+30]+'png')
    pass
