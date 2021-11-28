#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J.P Ballestrino, C. Deandraya, C. Uviedo
"""
import numpy as np

import os
import argparse
import time
import autobol
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
    ap.add_argument("-l","--list", type=str, required=True,help="Text file where input files are specified, one per line.")
    ap.add_argument("-m","--mode", type=str, required=True,help="predict, train or mask")
    ap.add_argument("-u", "--umbral", type=float, required=False, help="Umbral de desición, válido solo para predict ")
    ap.add_argument("-t", "--test_sz", type=float, required=False, help="Tamaño del conjunto test, válido solo para train ")
    ap.add_argument("-mod", "--model", type=str, required=False,  help="Modelo a usar, solo para predict ")
    #
    # INICIALIZACION
    #
    args = vars(ap.parse_args())
    indir = args["indir"]
    outdir = args["outdir"]
    list_file = args["list"]
    mode = args["mode"]
    umb=args["umbral"]
    test_sz=args["test_sz"]
    model = args["model"]
    #
    # open file list
    #
    with open(list_file) as fl:
        t0 = time.time()
        nfiles = 0
        #
        # process each file
        #
        if mode == "predict":
            for relfname in fl:
                nfiles += 1
                #
                relfname = relfname.rstrip('\n')
                reldir, fname = os.path.split(relfname)
                fbase, fext = os.path.splitext(fname)
                foutdir = outdir  # os.path.join(outdir,reldir)
                debugdir = os.path.join(foutdir, fbase + "_debug")

                print(f'\t{relfname}')

                if not os.path.exists(foutdir):  # create output dir
                    os.makedirs(foutdir)

                output_fname = os.path.join(foutdir, fname)
                if os.path.exists(output_fname):  # skip if already processed
                    continue

                input_fname = os.path.join(indir, relfname)
                if not os.path.exists(input_fname):
                    print('file not found')
                    continue
                #
                # ---------------------------------------------------
                # actual processing starts here
                # ---------------------------------------------------
                #
                autobol.predict(input_fname, foutdir,model,umb)
                #
                # ---------------------------------------------------
                #
        elif mode == "train":
            autobol.train(indir, outdir,test_sz)

        elif mode == "mask":
            autobol.crear_mascara(indir)
        else:
            print('unknown mode')
            exit(1)
            #
            # for each file
            #
    print('done')
    #
    # if main
    #

# -----------------------------------------------------------------------------
