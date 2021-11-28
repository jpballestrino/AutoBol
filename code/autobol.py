#-*- coding:utf-8 -*-
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
def predict(fname,salida,model,umbral=0.07):
    '''
    Predict
    '''
    #
    # Load Model
    #
    booster = xgb.Booster()
    booster.load_model(model)
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

def train(fname,fout,test_sz=0.15):

    print(fname,fout)
    df_data = pd.read_csv(fname)
    df_data = df_data.drop(labels='Unnamed: 0', axis=1, index=None, columns=None, level=None, inplace=False, errors='raise')
    y = df_data.pop('label').to_frame()

    X_train, X_test, y_train, y_test = train_test_split(df_data, y, stratify=y, random_state=30, test_size=test_sz)
    positive_instances = y_train['label'].sum()
    negative_instances = y_train['label'].count() - positive_instances
    Resp1 = 'Distinto'
    while (Resp1 != 'Y') and (Resp1 != 'y') and (Resp1 != 'N') and (Resp1 != 'n'):
        Texto1 = 'Cantidad de bólidos', positive_instances, 'Cantidad de no bólidos', negative_instances, 'Factor no bólidos/bólidos',negative_instances / positive_instances,'Desea continuar? Y/N'
        Resp1 = input(Texto1)
        if Resp1 == 'Y' or Resp1 == 'y':
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            params = {
                'colsample_bytree': 0.7,
                'eta': 0.3,
                'eval_metric': 'mae',
                'max_depth': 4,
                'min_child_weight': 0,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'scale_pos_weight': negative_instances / positive_instances
            }
            num_boost_round = 10000

            gridsearch_params = [
                (max_depth, min_child_weight)
                for max_depth in range(1, 10)
                for min_child_weight in range(0, 4)
            ]
            # Define initial best params and MAE
            min_mae = float("Inf")
            best_params = None
            for max_depth, min_child_weight in gridsearch_params:
                print("CV with max_depth={}, min_child_weight={}".format(
                    max_depth,
                    min_child_weight))
                # Update our parameters
                params['max_depth'] = max_depth
                params['min_child_weight'] = min_child_weight
                # Run CV
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics={'mae'},
                    early_stopping_rounds=20
                )
                # Update best MAE
                mean_mae = cv_results['test-mae-mean'].min()
                boost_rounds = cv_results['test-mae-mean'].argmin()
                print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
                if mean_mae < min_mae:
                    min_mae = mean_mae
                    best_params = (max_depth, min_child_weight)
            print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

            params['max_depth'] = best_params[0]
            params['min_child_weight'] = best_params[1]

            gridsearch_params = [
                (subsample, colsample)
                for subsample in [i / 10. for i in range(5, 11)]
                for colsample in [i / 10. for i in range(5, 11)]
            ]

            min_mae = float("Inf")
            best_params = None
            # We start by the largest values and go down to the smallest
            for subsample, colsample in reversed(gridsearch_params):
                print("CV with subsample={}, colsample={}".format(
                    subsample,
                    colsample))
                # We update our parameters
                params['subsample'] = subsample
                params['colsample_bytree'] = colsample
                # Run CV
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics={'mae'},
                    early_stopping_rounds=20
                )
                # Update best score
                mean_mae = cv_results['test-mae-mean'].min()
                boost_rounds = cv_results['test-mae-mean'].argmin()
                print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
                if mean_mae < min_mae:
                    min_mae = mean_mae
                    best_params = (subsample, colsample)
            print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

            params['subsample'] = best_params[0]
            params['colsample_bytree'] = best_params[1]

            # This can take some time…
            min_mae = float("Inf")
            best_params = None
            for eta in [1, 0.9, 0.8, 0.6, 0.5, .4, .3, .2, 0.1, 0.05]:
                print("CV with eta={}".format(eta))
                # We update our parameters
                params['eta'] = eta
                # Run and time CV
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics=['mae'],
                    early_stopping_rounds=10
                )
                # Update best score
                mean_mae = cv_results['test-mae-mean'].min()
                boost_rounds = cv_results['test-mae-mean'].argmin()
                print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
                if mean_mae < min_mae:
                    min_mae = mean_mae
                    best_params = eta
            print("Best params: {}, MAE: {}".format(best_params, min_mae))

            params['eta'] = best_params
            print(params)

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
            print('El mejor umbral según el f2_score es:',umbral)

            matrix=confusion_matrix(y_test, y_pred, labels=[0, 1])
            print(matrix)

            best_model.save_model("model_NUEVO.bin")


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
    if os.path.exists('Mascaras\Mascara_Station_' + str(station) + '.png'):

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
