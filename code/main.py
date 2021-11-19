#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: J.P Ballestrino, C. Deandraya, C. Uviedo
"""
import numpy
import math
# base packages
#
import os
import time
import argparse
#
# own modules
#
import autobol

#------------------------------------------------------------------------------

if __name__ == '__main__':
    #
    # ARGUMENTOS DE LINEA DE COMANDOS
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--indir", type=str, required=True,
		    help="path indir  where to find files")
    ap.add_argument("-o","--outdir", type=str,required=True, 
		    help="where to store results")
    ap.add_argument("-l","--list", type=str, required=True,
		    help="text file where input files are specified, one per line.")
    ap.add_argument("-m","--mode", type=str, required=True,
		    help="predict or train")
    #
    # INICIALIZACION
    #
    args   = vars(ap.parse_args())
    indir  = args["indir"]
    outdir = args["outdir"]
    list_file = args["list"]
    mode   = args["mode"]
    #
    # open file list
    #
    with open(list_file) as fl:
        t0 = time.time()
        nfiles = 0
        #
        # process each file
        #
        for relfname in fl:
            nfiles += 1 

            relfname     = relfname.rstrip('\n')
            reldir,fname = os.path.split(relfname)
            fbase,fext   = os.path.splitext(fname)
            foutdir      = outdir #os.path.join(outdir,reldir)
            debugdir     = os.path.join(foutdir,fbase + "_debug")            

            print(f'{nimage:8d}\t{relfname}')
            
            if not os.path.exists(foutdir): # create output dir
                os.makedirs(foutdir)

            output_fname = os.path.join(foutdir,fname)
            if os.path.exists(output_fname): # skip if already processed
                continue

            input_fname = os.path.join(indir,relfname)
            if not os.path.exists(input_fname):
                print('file not found')
                continue
            #
            #---------------------------------------------------
            # actual processing starts here
            #---------------------------------------------------
            #
            if args.mode == "predict":
                autobol.predict()
            elif args.mode == "train":
                autobol.train()
            else:
                print('unknown mode')
                exit(1)

            

            #
            #---------------------------------------------------
            #
        #
        # for each file
        #
    print('done')
    #
    # if main
    #
    
#-----------------------------------------------------------------------------
