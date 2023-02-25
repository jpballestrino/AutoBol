#!/usr/bin/env python3

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
import argparse
import time
import functions
from pyfiglet import Figlet


if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    f = Figlet(font='starwars')
    print(f.renderText('Autobol'))

    ap = argparse.ArgumentParser(description='This implementation allows you to create a new mask, clasiffy or re-train the model')
    ap.add_argument("-i", "--indir", type=str, required=True,help="path dir where to find files")
    ap.add_argument("-o","--outdir", type=str,required=True,help="Where to store results")
    ap.add_argument("-m","--mode", type=str, required=True,help="predict, train or mask")
    ap.add_argument("-u", "--umbral", type=float, required=False, help="Umbral de desición, válido solo para predict ")
    ap.add_argument("-t", "--test_sz", type=float, required=False, help="Tamaño del conjunto test, válido solo para train ")
    ap.add_argument("-mod", "--model", type=str, required=False,  help="Modelo a usar, solo para predict ")
    ap.add_argument("-verbose",action="store_true")
    ap.add_argument("-localize", action="store_true")
    #
    # INICIALIZACION
    #
    args = vars(ap.parse_args())
    indir = args["indir"]
    outdir = args["outdir"]
    mode = args["mode"]
    umb=args["umbral"]
    test_sz=args["test_sz"]
    model = args["model"]
    verbose= args["verbose"]
    loc=args["localize"]

    #
    # open file list
    #

    t0 = time.time()
    nfiles = 0
        #
        # process each file
        #

    if mode == "predict":
        if verbose:
            foutdir = outdir
            for dirpath, dirnames, filenames in os.walk(indir):
                for filename in filenames:
                    filename = os.fsdecode(filename)
                    if filename.endswith(".avi"):
                        pathh= dirpath+'/' + os.path.join(filename)
                        print(pathh)
                        functions.predict(pathh, foutdir,model,umb,True,False)
        else:
            if loc:
                foutdir = outdir
                for dirpath, dirnames, filenames in os.walk(indir):
                    for filename in filenames:
                        filename = os.fsdecode(filename)
                        if filename.endswith(".avi"):
                            pathh = dirpath + '/' + os.path.join(filename)
                            print(pathh)
                            functions.predict(pathh, foutdir,model,umb,False,True)
            else:
                foutdir = outdir
                for dirpath, dirnames, filenames in os.walk(indir):
                    for filename in filenames:
                        filename = os.fsdecode(filename)
                        if filename.endswith(".avi"):
                            pathh = dirpath + '/' + os.path.join(filename)
                            print(pathh)
                            functions.predict(pathh, foutdir, model, umb, False, False)

    elif mode == "train":
        functions.train(indir, outdir,test_sz)

    elif mode == "mask":
        functions.crear_mascara(indir)
    else:
        print('unknown mode')
        exit(1)

    print('done')


# -----------------------------------------------------------------------------