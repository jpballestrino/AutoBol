#-*- coding:utf-8 -*-

import xgboost as xgb
import algoritmos
import pandas as pd
import numpy as np
import os
def predict(fname,salida,umbral=0.07):
    '''
    Predict
    '''
    #
    # Load Model
    #
    booster = xgb.Booster()
    booster.load_model('model.bin')
    data=[]
    velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int, var_int, var_vel, mean_fwhm, \
    var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1, V_feat_fwhm_2, loss, d = algoritmos.Clasificar(fname)
    data.append([velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int, var_int, var_vel,
         mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1, V_feat_fwhm_2, loss, d])
    df = pd.DataFrame(data,columns=['velocidad media', 'mean_intensidad', 'N', 'dispersion', 'max_vel', 'max_int', 'min_vel',
                               'min_int', 'var_int', 'var_vel', 'mean_fwhm', 'var_fwhm', 'V_feat_i_1', 'V_feat_i_2',
                               'V_feat_v_1', 'V_feat_v_2', 'V_feat_fwhm_1', 'V_feat_fwhm_2', 'loss', 'recorrido' ])
    df['V_feat_fwhm_1'] = df['V_feat_fwhm_1'].replace([np.nan, np.inf, -np.inf], 0)
    df['V_feat_fwhm_2'] = df['V_feat_fwhm_2'].replace([np.nan, np.inf, -np.inf], 0)
    df['V_feat_v_1'] = df['V_feat_v_1'].replace([np.nan, np.inf, -np.inf], 0)
    df['V_feat_v_2'] = df['V_feat_v_2'].replace([np.nan, np.inf, -np.inf], 0)
    df['V_feat_i_1'] = df['V_feat_i_1'].replace([np.nan, np.inf, -np.inf], 0)
    df['V_feat_i_2'] = df['V_feat_i_2'].replace([np.nan, np.inf, -np.inf], 0)
    df['var_fwhm'] = df['var_fwhm'].replace([np.nan, np.inf, -np.inf], 0)
    df['mean_fwhm'] = df['mean_fwhm'].replace([np.nan, np.inf, -np.inf], 0)
    df['recorrido'] = df['recorrido'].replace([np.nan, np.inf, -np.inf], 0)

    dpredict = xgb.DMatrix(df)

    prob=booster.predict(dpredict)
    log=[]
    porcentaje=np.round(prob*100,decimals=2)[0]
    if (prob > umbral):
        log.append([fname, porcentaje, 'Bólido'])
    else:
        log.append([fname, porcentaje, 'No Bólido'])
    prediccion = pd.DataFrame(log, columns=['Nombre Archivo', 'Probabilidad de Bólido', 'Etiqueta'])

    if not os.path.exists(salida +'\Predicciones.csv'):  # create output di
        prediccion.to_csv(salida+ '\Predicciones.csv', index=False)
    else:
        prediccion.to_csv(salida+'\Predicciones.csv', mode='a', index=False, header=False)



    pass

def train():


    print('entra 2')

    pass

def crear_mascara(fname):

    global filll
    global colll
    global counter
    fill = []
    coll = []
    counter = -1

    station = algoritmos.numero_estacion(fname)
    if os.path.exists('Mascaras\Mascara_Station_' + str(station) + '.png'):
        #print()
        Cont=0
        Resp1='Distinto'
        while (Resp1!='Y') and (Resp1!='y') and (Resp1!='N') and (Resp1!='n') and(Cont<3):
            Cont=Cont+1
            Texto1='La máscara de la estación ' + str(station)+' ya existe, desea recrearla? Y or N'
            Resp1 = input(Texto1)
            if Resp1=='Y' or Resp1=='y':
                algoritmos.CrearMascara(fname)

            else:
                pass

        if (Resp1!='Y')and(Resp1!='y')and(Resp1!='N')and(Resp1!='n'):
            print('Ejecute de nuevo el modulo con una respuesta valida')

    else:
        algoritmos.CrearMascara(fname)

    pass
