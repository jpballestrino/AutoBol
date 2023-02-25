# -*- coding:utf-8 -*-
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

import xgboost as xgb
import algoritmos
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, confusion_matrix
import abio
import time
from algoritmos import get_frames


def predict(fname, salida, model, umbral=0.083, verbose=False, localize=False):
    '''
    Predict
    '''

    if verbose:
        data = []
        ruta_entrada, velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int, var_int, var_vel, mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1, V_feat_fwhm_2, loss, d = algoritmos.Clasificar(
            fname)
        data.append(
            [ruta_entrada, velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int,
             var_int, var_vel, mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1,
             V_feat_fwhm_2, loss, d, 'NaN'])
        data_ft = pd.DataFrame(data,
                               columns=['Video', 'velocidad media', 'mean_intensidad', 'N', 'dispersion', 'max_vel',
                                        'max_int', 'min_vel', 'min_int', 'var_int', 'var_vel', 'mean_fwhm', 'var_fwhm',
                                        'V_feat_i_1', 'V_feat_i_2', 'V_feat_v_1', 'V_feat_v_2', 'V_feat_fwhm_1',
                                        'V_feat_fwhm_2', 'loss', 'recorrido', 'Clasificación'])
        abio.save_predict(data_ft, salida, verbose)

    else:
        booster = xgb.Booster()
        booster.load_model(model)
        data = []
        ruta_entrada, velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int, var_int, var_vel, mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1, V_feat_fwhm_2, loss, d = algoritmos.Clasificar(
            fname)
        data.append(
            [ruta_entrada, velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int,
             var_int, var_vel, mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1,
             V_feat_fwhm_2, loss, d])
        df = pd.DataFrame(data, columns=['Video', 'velocidad media', 'mean_intensidad', 'N', 'dispersion', 'max_vel',
                                         'max_int', 'min_vel', 'min_int', 'var_int', 'var_vel', 'mean_fwhm', 'var_fwhm',
                                         'V_feat_i_1', 'V_feat_i_2', 'V_feat_v_1', 'V_feat_v_2', 'V_feat_fwhm_1',
                                         'V_feat_fwhm_2', 'loss', 'recorrido'])
        df['V_feat_fwhm_1'] = df['V_feat_fwhm_1'].replace([np.nan, np.inf, -np.inf], 0)
        df['V_feat_fwhm_2'] = df['V_feat_fwhm_2'].replace([np.nan, np.inf, -np.inf], 0)
        df['V_feat_v_1'] = df['V_feat_v_1'].replace([np.nan, np.inf, -np.inf], 0)
        df['V_feat_v_2'] = df['V_feat_v_2'].replace([np.nan, np.inf, -np.inf], 0)
        df['V_feat_i_1'] = df['V_feat_i_1'].replace([np.nan, np.inf, -np.inf], 0)
        df['V_feat_i_2'] = df['V_feat_i_2'].replace([np.nan, np.inf, -np.inf], 0)
        df['var_fwhm'] = df['var_fwhm'].replace([np.nan, np.inf, -np.inf], 0)
        df['mean_fwhm'] = df['mean_fwhm'].replace([np.nan, np.inf, -np.inf], 0)
        df['recorrido'] = df['recorrido'].replace([np.nan, np.inf, -np.inf], 0)
        df = df.drop(labels='Video', axis=1, index=None, columns=None, level=None, inplace=False, errors='raise')
        dpredict = xgb.DMatrix(df)
        prob = booster.predict(dpredict)
        log = []
        porcentaje = np.round(prob * 100, decimals=2)[0]
        if (prob > umbral):
            log.append([fname, porcentaje, 'Bólido'])
        else:
            log.append([fname, porcentaje, 'No Bólido'])
        prediccion = pd.DataFrame(log, columns=['Nombre Archivo', 'Probabilidad de Bólido', 'Etiqueta'])
        abio.save_predict(prediccion, salida, verbose)
        if localize:
            if N > 0:
                if (prob > umbral):
                    framesarray, fps = get_frames(fname)
                    im1, i = algoritmos.localizer(fname, framesarray, salida)
                    abio.save_img(im1, fname, i, salida)
    pass


def train(fname, fout, test_sz=0.15):
    inicio = time.time()
    print(fname, fout)
    df_data = pd.read_csv(fname)
    df_data = df_data.dropna(how='any')
    aux = []
    aux.append(
        ['auxiliar', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0])
    aux = pd.DataFrame(aux,
                       columns=['Video', 'velocidad media', 'mean_intensidad', 'N', 'dispersion', 'max_vel', 'max_int',
                                'min_vel', 'min_int', 'var_int', 'var_vel', 'mean_fwhm', 'var_fwhm', 'V_feat_i_1',
                                'V_feat_i_2', 'V_feat_v_1', 'V_feat_v_2', 'V_feat_fwhm_1', 'V_feat_fwhm_2', 'loss',
                                'recorrido', 'label'])

    df_data = df_data.drop(['var_vel'], axis=1)
    df_data = df_data.drop(['mean_fwhm'], axis=1)
    df_data = df_data.drop(['var_fwhm'], axis=1)

    df_data = df_data.loc[(df_data != 0.0).any(axis=1)]
    df_data = pd.concat([df_data, aux], axis=0)
    df_data = df_data.drop(labels='Video', axis=1)
    y = df_data.pop('label').to_frame()

    X_train, X_test, y_train, y_test = train_test_split(df_data, y, stratify=y, random_state=30, test_size=test_sz)
    positive_instances = y_train['label'].sum()
    negative_instances = y_train['label'].count() - positive_instances
    positive_test = y_test['label'].sum()
    negative_test = y_test['label'].count() - positive_test
    Resp1 = 'Distinto'
    while (Resp1 != 'Y') and (Resp1 != 'y') and (Resp1 != 'N') and (Resp1 != 'n'):
        Texto1 = 'Cantidad de bólidos en entrenamiento', positive_instances, 'Cantidad de no bólidos en entrenamiento', negative_instances, 'Cantidad de bolidos en test', positive_test, 'Cantidad de no bolidos en test', negative_test, 'Desea continuar? Y/N'
        Resp1 = input(Texto1)
        if Resp1 == 'Y' or Resp1 == 'y':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {
                'colsample_bytree': 0.8,
                'eta': 0.3,
                'eval_metric': 'mae',
                'max_depth': 4,
                'min_child_weight': 0,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'scale_pos_weight': negative_instances / positive_instances
            }
            num_boost_round = 3000

            gridsearch_params = [
                (max_depth, subsample, min_child_weight, eta)
                for max_depth in range(3, 7)
                for subsample in [i / 10. for i in range(5, 9)]
                for min_child_weight in range(5, 9)
                for eta in [.1, .2, 0.3]

            ]
            # Define initial best params and MAE
            min_mae = float("Inf")
            best_params = None
            for max_depth, subsample, min_child_weight, eta in gridsearch_params:
                print("CV with max_depth={},subsample={}, min_child_weight={}, eta={} ".format(
                    max_depth,
                    subsample,
                    min_child_weight,
                    eta,
                ), 'current min mae', min_mae)
                # Update our parameters
                params['max_depth'] = max_depth
                params['subsample'] = subsample
                params['min_child_weight'] = min_child_weight
                params['eta'] = eta

                # Run CV
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics='mae',
                    early_stopping_rounds=20
                )
                # Update best MAE
                mean_mae = cv_results['test-mae-mean'].min()
                boost_rounds = cv_results['test-mae-mean'].argmin()
                print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
                if mean_mae < min_mae:
                    min_mae = mean_mae
                    best_params = (max_depth, subsample, min_child_weight, eta)
            print("Best params: {},{},{}, {}, MAE: {}".format(best_params[0], best_params[1], best_params[2],
                                                              best_params[3], min_mae))

            params['max_depth'] = best_params[0]
            params['subsample'] = best_params[1]
            params['min_child_weight'] = best_params[2]
            params['eta'] = best_params[3]

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "Test")],
                early_stopping_rounds=20
            )

            num_boost_round = model.best_iteration + 1
            best_model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtest, "Test")]
            )

            umbrales = np.linspace(0, 1, 1000)
            f2 = []
            for umbral in umbrales:
                y_pred = (best_model.predict(dtest) > umbral) * 1
                f2.append(fbeta_score(y_test, y_pred, beta=2))
            umbral = np.argmax(f2) / 1000
            y_pred = (best_model.predict(dtest) > umbral) * 1
            print('El mejor umbral según el f2_score es:', umbral)

            matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
            print(params)
            print(matrix)

            best_model.save_model("model_" + str(umbral) + ".bin")
            f = open("model_" + str(umbral) + ".txt", "w")
            f.write("Informacion de model_" + str(umbral) + ".bin entrenado utilizando xgboost: \n")
            f.write('Modelo entrenado con: ' + str(positive_instances) + ' bólidos y ' + str(
                negative_instances) + ' no bólidos en entrenamiento y luego testeado con ' + str(
                positive_test) + ' bolidos y ' + str(negative_test) + ' de no bolidos en test\n')
            f.write('El mejor umbral encontrado según el f2_score es:  \n')
            f.write(str(umbral) + "\n")
            f.write("Con los hiperparametros: \n ")
            str_params = repr(params)
            f.write(str_params + "\n")
            f.write("y matriz de confusion: \n ")
            str_matrix = repr(matrix)
            f.write(str_matrix)
            f.write('\n')
            sorted_idx = best_model.get_score(importance_type='weight')
            f.write("feature importance: \n ")
            f.write(str(sorted_idx))
            f.write('\n')
            fin = time.time()
            elapsed = fin - inicio
            f.write('tiempo de entrenamiento total ' + str(elapsed) + ' segundos')
            f.close()




        else:
            pass

    pass


def crear_mascara(fname):
    global filll
    global colll
    global counter
    fill = []
    coll = []
    counter = -1

    station = algoritmos.numero_estacion(fname)
    if os.path.exists('./Mascaras/Mascara_Station_' + str(station) + '.png'):

        Cont = 0
        Resp1 = 'Distinto'
        while (Resp1 != 'Y') and (Resp1 != 'y') and (Resp1 != 'N') and (Resp1 != 'n') and (Cont < 3):
            Cont = Cont + 1
            Texto1 = 'La máscara de la estación ' + str(station) + ' ya existe, desea recrearla? Y or N'
            Resp1 = input(Texto1)
            if Resp1 == 'Y' or Resp1 == 'y':
                algoritmos.CrearMascara(fname)

            else:
                pass

        if (Resp1 != 'Y') and (Resp1 != 'y') and (Resp1 != 'N') and (Resp1 != 'n'):
            print('Ejecut-e de nuevo el modulo con una respuesta valida')

    else:
        algoritmos.CrearMascara(fname)

    pass