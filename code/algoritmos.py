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

import numpy
import math
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy import signal
from moviepy.editor import VideoFileClip
import os
import cv2
import skimage.draw
import matplotlib.image as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import re
import pandas as pd
import abio
import algoritmos

def localizer(fname,framesarray,fout):
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
    abio.save_predict(data_ft, fout, True)

    x, y, excluidos_c_RANSAC,excluidos_f_RANSAC,excluidos_c_filtro,excluidos_f_filtro,xc,yc,r = tracking(fname,True)
    maximo = numpy.percentile(framesarray, 100, axis=0)
    max_x=numpy.max(x)+3
    min_x=numpy.min(x)-3

    max_y=numpy.max(y)+3
    min_y=numpy.min(y)-3
    coincide=False
    i=0
    theta = numpy.linspace(0, 2 * numpy.pi, 100)
    x1 = r * numpy.cos(theta) + xc
    x2 = r * numpy.sin(theta) + yc
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.plot(x1, x2)  # Circunferencia

    ax.plot(excluidos_c_RANSAC, excluidos_f_RANSAC, '*b', label='outliers RANSAC')
    ax.plot(excluidos_c_filtro, excluidos_f_filtro, '*g', label='outliers filtro')
    ax.plot(x, y, '*y', label='inlier 2da vuelta con filtro')
    rect=patches.Rectangle((min_x,min_y),(max_x-min_x) , (max_y-min_y), linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(fname)
    ax.set_aspect(1)
    ax.imshow(maximo,cmap="gray")
    plt.legend()

    # Regular expression pattern to extract the station number and timestamp without the extension
    pattern = r"Station_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
    # Using regular expression to extract the station number and timestamp
    match = re.search(pattern, fname)
    if match:
        station_number = match.group(1)
        timestamp = match.group(2)
        result = f"Station_{station_number}_{timestamp}."
    else:
        result="NombreDelVideoNoPudoSerExtraido"
    plt.savefig(fout + '/' + result )  # Change the file name and extension as needed

    return

def tracking(ruta_entrada, ploteo=False):
    """
    	If it exists, the event that occurs during the video is tracked, throughout the frames.
    	In addition to the determination of some of its characteristics.

    	Inputs:
    		- ruta_entrada: String, with the location and name of the file to execute.
    		- ploteo: Booleano, allows plotting RANSAC model.

    	Returns:
    		- ruta_entrada: String, with the location and name of the file to execute.
    		- tiempo: A numpy array, with the time vector of the video.
    		- X_model: A numpy array, with the horizontal position of the event in the video.
    		- Y_model: A numpy array, with the vertical position of the event in the video.
    		- LC: A numpy array, with the intensity value of the object relative to the background intensity.
    		- N: Integer, with the value of the size of the numpy array that contains the frames of the video.
    		- total_dispersion: Integer, with the value of the average dispersion of the event positions in
    		the frames.
    		- FWHMarray: A numpy array, with the value of the FWHM of the event in the frames.
    		- loss: Integer, with the mean square error between the trajectory it makes and the model of a
    		circumference.
    		- d: Integer, with the value of the distance traveled.
    	"""

    framesarray, fps = get_frames(ruta_entrada)
    video_tracking = numpy.empty_like(framesarray)  # Creo video vacio para luego cargar el tracking
    cantidad_frames, cantidad_filas, cantidad_columnas = framesarray.shape
    fondo = numpy.percentile(framesarray, 5, axis=0)  # Obtengo la imagen fondo
    maximo = numpy.percentile(framesarray, 95, axis=0)  # Obtengo la imagen maximo
    fila = numpy.empty(cantidad_frames)  # Creo el vector donde estaran las coordenadas fila del pixel con mayor intensidad
    columna = numpy.empty(cantidad_frames)  # Creo el vector donde estaran las coordenadas columnas del pixel con mayor intensidad
    #
    # se calcula el centro de masas para todos los frames en los que se detecto objeto dentro de la ventana
    #
    ventana = 6
    X_detectadas = []
    Y_detectadas = []
    #
    # CARGO LA MASCARA
    #
    station = numero_estacion(ruta_entrada)
    if not os.path.exists('./Mascaras/Mascara_Station_' + str(station) + '.png'):
        print('La máscara de la estación ',station,' no existe, creela ahora!')
        global filll
        global colll
        global counter
        filll=[]
        colll=[]
        counter=-1
        CrearMascara(ruta_entrada)

    Mask = plt.imread('./Mascaras/Mascara_Station_' + str(station) + '.png')  # RUTA DE LAS MASCARAS
    Mask = numpy.asarray(Mask)[:, :, 0]
    #
    # obtengo las coordenadas del pixel máximo frame a frame
    #
    for i in range(cantidad_frames):
        resta = numpy.abs(framesarray[i] - fondo)
        video_tracking[i] = resta
        _, fila[i], columna[i] = numpy.unravel_index(numpy.argmax(video_tracking[i] * Mask),
                                                     video_tracking.shape)  # coordenadas del máximo dentro del frame enmascarado
    img = maximo * Mask
    aux = img[img > 0]
    umbral = numpy.percentile(aux,20)  # Defino el umbral a utilizar, solo tomando en cuenta la parte no nula luego de enmascarar
    intervalo = 1 / fps
    for i in range(cantidad_frames):
        if video_tracking[i, int(fila[i]), int(columna[
                                                   i])] > umbral:  # Si la coordenada mas brillante supera cierto umbral definido entonces se guardan las coordenadas del objeto
            x, y = center_mass(video_tracking[i], fila[i], columna[i], ventana)
            X_detectadas.append(x)
            Y_detectadas.append(y)

    X_detectadas = numpy.asarray(X_detectadas).reshape(-1, 1)  # paso las coordenadas del objeto a numpy
    Y_detectadas = numpy.asarray(Y_detectadas).reshape(-1, 1)
    if (len(X_detectadas) >= 3):
        pocket = RANSAC_3(X_detectadas, Y_detectadas, residual_treshold=5)
        xc = pocket[0][0][0]
        yc = pocket[0][1][0]
        r = pocket[0][2][0]
        pertenecen = pocket[0][3]
        loss = pocket[0][4]
        no_pertenecen = numpy.logical_not(pertenecen)
        excluidos_c_RANSAC = X_detectadas[no_pertenecen]
        excluidos_f_RANSAC = Y_detectadas[no_pertenecen]

        # -----------------------------------------------------------------------------------------------------------

        umbral2 = numpy.percentile(aux, 5)  # Defino el umbral a utilizar segunda vuelta
        X_model = []
        Y_model = []
        tiempo = []
        pos_frame_abs = []
        pertenece_model = []
        for i in range(cantidad_frames):

            if video_tracking[i, int(fila[i]), int(columna[i])] > umbral2 and abs(
                    numpy.sqrt((columna[i] - xc) ** 2 + (fila[i] - yc) ** 2) - r) <= 5:
                x2, y2 = center_mass(video_tracking[i], fila[i], columna[i], ventana)
                if (numpy.isnan(x2)) or (numpy.isnan(y2)):
                    pass
                else:
                    X_model.append(x2)
                    Y_model.append(y2)
                    tiempo.append(intervalo * i)
                    pos_frame_abs.append(i)
                    pertenece_model.append(True)

        X_model = numpy.asarray(X_model).reshape(-1, 1)  # paso las coordenadas del objeto a numpy
        Y_model = numpy.asarray(Y_model).reshape(-1, 1)
        pos_frame_abs = numpy.asarray(pos_frame_abs).reshape(-1, 1)
        tiempo = numpy.asarray(tiempo).reshape(-1, 1)
        pertenece_model = filtro_extra(X_model, Y_model, pertenece_model)
        no_pertenecen2 = numpy.logical_not(pertenece_model)
        excluidos_c_filtro = X_model[no_pertenecen2]
        excluidos_f_filtro = Y_model[no_pertenecen2]
        X_model = X_model[pertenece_model]
        Y_model = Y_model[pertenece_model]
        pos_frame_abs = pos_frame_abs[pertenece_model]
        tiempo = tiempo[pertenece_model]

        d = distancia_recorrida(r, ang([[xc, yc], [X_model[0], Y_model[0]]], [[xc, yc], [X_model[-1], Y_model[-1]]]))

        # -----------------------------------------------------------------------------------------------------------

        FWHMarray = []
        LC = []
        for p in range(len(pos_frame_abs)):
            if framesarray[pos_frame_abs[p][0]][round(Y_model[p][0]), round(X_model[p][0])] == 255:
                # entonces tengo que hacer reconstrucción
                mar = 5
                Z = zoom_on_frame(framesarray[pos_frame_abs[p][0]], round(X_model[p][0]), round(Y_model[p][0]),
                                  margin=mar)
                saturated = numpy.equal(Z, 255)
                Zrec = reconstruir_brillo(Z, saturated)
                if (numpy.max(Zrec) > 255):
                    framesarray[pos_frame_abs[p][0]][round(Y_model[p][0]) - mar:round(Y_model[p][0]) + mar,
                    round(X_model[p][0]) - mar:round(X_model[p][0]) + mar] = Zrec

            fwhm, fwhm_sp = FWHM(video_tracking[pos_frame_abs[p]], Y_model[p][0], X_model[p][0])
            FWHMarray.append(fwhm_sp)
            lc = CurvaDeLuz(framesarray[pos_frame_abs[p]], fondo, Y_model[p][0], X_model[p][0], 2 * round(fwhm_sp))
            LC.append(lc)
        total_dispersion = dispersion(X_model, Y_model)

        # Grafica la recta obtenida en RANSAC
        if ploteo:
            print("Ploteando resultados de RANSAC...")
            theta = numpy.linspace(0, 2 * numpy.pi, 100)
            x1 = r * numpy.cos(theta) + xc
            x2 = r * numpy.sin(theta) + yc
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.plot(x1, x2)  # Circunferencia

            ax.plot(excluidos_c_RANSAC, excluidos_f_RANSAC, '*b', label='outliers RANSAC')
            ax.plot(excluidos_c_filtro, excluidos_f_filtro, '*g', label='outliers filtro')
            ax.plot(X_model, Y_model, '*y', label='inlier 2da vuelta con filtro')
            ax.set_title(ruta_entrada)
            ax.set_aspect(1)
            ax.imshow(maximo, cmap='gray')
            plt.legend()
            return X_model, Y_model, excluidos_c_RANSAC,excluidos_f_RANSAC,excluidos_c_filtro,excluidos_f_filtro,xc,yc,r

        return ruta_entrada,tiempo, X_model, Y_model, LC, len(X_model), total_dispersion[0], FWHMarray, loss, d

    else:
        if (len(X_detectadas) < 3):
            print('Evento con pocos puntos (<3), verificar si la máscara no borra el evento.')
        return ruta_entrada, [numpy.nan], [numpy.nan], [numpy.nan], [numpy.nan], numpy.nan, numpy.nan, [
            numpy.nan], numpy.nan, numpy.nan


# -----------------------------------------------------------------------------------------------------------''
def distancia_recorrida(r, ang):
    """
    Measures the arc distance between the first and last point of the event.

    Inputs:
    	- r: Integer, with the value of the radius of the circumference model described by the event.
    	- ang: Integer with the value of the circumference angle traveled described by the event.

    Returns:
    	- d: Integer, with the value of the distance traveled.
    """
    dist_recorrida = r * ang
    return dist_recorrida


# -------------------------------------------------------------------------------------


def dot(vA, vB):
    """
    Get dot prod.

    Inputs:
    	- vA: A numpy array 1x2
    	- vB: A numpy array 1x2

    Returns:
    	- Integer, with the value vA[0] * vB[0] + vA[1] * vB[1]
    """
    return vA[0] * vB[0] + vA[1] * vB[1]


def ang(lineA, lineB):
    """
    Calculates the angle of an arc circumference.

    Inputs:
    	- lineA: A numpy array, with the position of the center of the circumference and the
    	initial position.
    	- lineB: A numpy array, with the position of the center of the circumference and the
    	final position.

    Returns:
    	- ang: Integer with the value of the circumference angle traveled described by the event.
    """
    # Get nicer vector form
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5
    # Get cosine value
    cos_ = dot_prod / magA / magB
    # Get angle in radians and then convert to degrees

    a = dot_prod / magB / magA
    if (a >= 1):
        angle = math.acos(1)
    else:
        angle = math.acos(a)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:
        # As in if statement
        return (360 - ang_deg) * numpy.pi / 180
    else:

        return ang_deg * numpy.pi / 180


# -------------------------------------------------------------------------------------
def velocidad(t, x, y):  # Calcula la velocidad en pixeles/sec
    """
    Calculate the instantaneous velocity of the event along the frames.

    Inputs:
    	- t: A numpy array, with the time vector of the video.
    	- x: A numpy array, with the horizontal position of the event in the video.
    	- y: A numpy array, with the vertical position of the event in the video.

    Returns:
    	- v: A numpy list, with the with the instantaneous speed of the event along the frames.
    """
    v = []
    for i in range(len(t)):
        if (i == 0):
            a = numpy.array([x[i][0], y[i][0]])
            b = numpy.array([x[i + 1][0], y[i + 1][0]])
            t1 = t[i]
            t2 = t[i + 1]
        else:
            a = numpy.array([x[i][0], y[i][0]])
            b = numpy.array([x[i - 1][0], y[i - 1][0]])
            t2 = t[i]
            t1 = t[i - 1]

        dist = numpy.linalg.norm(a - b)
        v.append(dist / (t2[0] - t1[0]))
    return v


# --------------------------------------------------------------------------------
def get_frames(ruta_entrada):
    """
    Gets the frames of a video from a file route and the fps of it.

    Inputs:
    	- ruta_entrada: String, with the location and name of the file to execute.

    Returns:
    	- frames: Numpy array, containing numpy array with video frames.
    	- video.fps: Integer, with the value of the frames per second of the video.
    """

    video = VideoFileClip(ruta_entrada)  # Cargo el video
    frames = []
    for i in video.iter_frames():  # Obtengo los frames
        frames.append(i[:, :, 0])
    frames = numpy.asarray(frames, dtype=float)  # Convierto los frames de lista a numpy
    return frames, video.fps


# -------------------------------------------------------------------------------------
def RANSAC_3(x, y, residual_treshold, ploteo=False):
    """
    Obtains the circumference model that best fits the N number of points.

    Inputs:
    	- x: A numpy array, with the horizontal position of the event in the video.
    	- y: A numpy array, with the vertical position of the event in the video.
    	- residual_treshold: Threshold for differential inliers from outliers.
    	- ploteo: Booleano, allows plotting RANSAC model.

    Returns:
    	- pocket: Numpy list, containing numpy array with the integer coordinates of
    	the center, the radius of the circumference. The inliners and the cumulative
    	sum of the difference between the points and the model.
    """
    actual_loss = 50000
    k = 0
    retorno = []
    retorno2 = []
    while k < 900 and actual_loss > len(x) / 4:
        i = numpy.random.randint(0, len(x), 3)
        xs = x[i]
        ys = y[i]

        # 2- Ajustar modelo
        x0 = xs[0]
        y0 = ys[0]
        x1 = xs[1]
        y1 = ys[1]
        x2 = xs[2]
        y2 = ys[2]
        a = (y1 - y0) / (x1 - x0)
        b = y1 - a * x1

        if ((a * x2 + b - y2) == 0):
            if (k == 899) and (not (bool(retorno))) and (not (bool(retorno2))):
                pocket = []
                pocket.append([numpy.array([x[0]]), numpy.array([y[0]]), numpy.array([3]), [False] * len(x), 10000])
                k = k + 1
                return pocket
            else:
                k = k + 1
        else:
            # r^2 = (x-x_0)^2+ (y-y_0)^2 ----> r^2= x^2-2x x_0+x_0^2+y^2+2y y_0 + y_0^2
            # puntos medios de los segmentos 0-1 y 1-2
            a = (x0 + x1) / 2
            b = (y0 + y1) / 2
            c = (x1 + x2) / 2
            d = (y1 + y2) / 2

            # rectas perpendiculares por (a,b) y (c,d)
            xp1 = 100
            xp2 = 100

            # interseccion: centro
            A1, B1, C1 = x1 - a, y1 - b, a * (x1 - a) + b * (y1 - b)
            A2, B2, C2 = x1 - c, y1 - d, c * (x1 - c) + d * (y1 - d)
            xc = (C2 - B2 * C1 / B1) / (A2 - B2 * A1 / B1)
            yc = (C1 - A1 * xc) / B1
            r = numpy.sqrt((x1 - xc) ** 2 + (y1 - yc) ** 2)
            if (numpy.isnan(xc)) or (numpy.isinf(xc)) or (r < 30):
                k = k + 1
                if (k == 900) and (numpy.isnan(r)) and not (bool(retorno)) and not (bool(retorno2)):
                    pocket = []
                    pocket.append([numpy.array([x[0]]), numpy.array([y[0]]), numpy.array([3]), [False] * len(x), 10000])
                    return pocket
                b = 9
                if not (bool(retorno)) and (r < 30):
                    loss = []
                    for i in range(len(x)):
                        diferencia = numpy.abs(distance.euclidean([x[i], y[i]], [xc, yc]) - r)
                        loss.append(diferencia)
                    if (numpy.sum(loss) < actual_loss):
                        MAD = stats.median_absolute_deviation(loss)
                        inliers = []
                        for i in range(len(loss)):
                            if ((numpy.abs(loss[i])) > residual_treshold * MAD):
                                inliers.append(False)
                            else:
                                inliers.append(True)
                        pocket = []
                        actual_loss = 30000
                        pocket.append([xc, yc, r, inliers, actual_loss])
                        retorno2.append(1)
            else:
                # 3- Testear contra los otros datos y medir la calidad
                loss = []
                for i in range(len(x)):
                    diferencia = numpy.abs(distance.euclidean([x[i], y[i]], [xc, yc]) - r)
                    loss.append(diferencia)
                if (numpy.sum(loss) < actual_loss):
                    MAD = stats.median_absolute_deviation(loss)
                    inliers = []
                    for i in range(len(loss)):
                        if ((numpy.abs(loss[i])) > residual_treshold * MAD):
                            inliers.append(False)
                        else:
                            inliers.append(True)
                    pocket = []
                    actual_loss = numpy.sum(loss)
                    pocket.append([xc, yc, r, inliers, actual_loss])
                    retorno.append(1)
                k = k + 1
    return pocket

# ----------------------------------------------------------------------------------------

def V_feature(array):
    """
    'Smoothness' of a feature.

    Inputs:
    	- array: A numpy array.

    Returns:
    	- feature1: Integer, with the sum of the distance between the points of a curve
    	resampled at 25 (fps).
    	- feature2: Integer, with the sum of the distance between the points of a curve
    	resampled at 25 (fps) and normalized.
    """
    M = 25
    N = len(array)
    resample = signal.resample(array, M)
    feature1 = 0
    feature2 = 0
    for k in range(1, M):
        feature1 = feature1 + numpy.abs(resample[k] - resample[k - 1])
    for i in range(1, N):
        feature2 = feature2 + numpy.abs(resample[k] - resample[k - 1])
    return feature1, feature2 / N


# ----------------------------------------------------------------------------------------



def numero_estacion(file_name):
    """
    Gets the station number from the file name.

    Inputs:
    	- file: String, with file name.

    Returns:
    	- f: Integer with corresponding station number.
    """

    # Regular expression pattern to extract the station number
    pattern = r"Station_(\d+)_"

    # Using regular expression to extract the station number
    match = re.search(pattern, file_name)

    if match:
        station_number = int(match.group(1))
        return station_number
    else:
        print('station number not found')


# ----------------------------------------------------------------------------------------

def MousePoints(event, x, y, flags, params):
    """
    Get points from the mouse and store them in lists.

    Inputs:
    	- event: Mouse button identifier code.
    	- x: Numpy list, containing numpy array with the horizontal position of the event in the video.
    	- y: Numpy list, containing numpy array with the vertical position of the event in the video.
    """

    global counter

    if event == cv2.EVENT_LBUTTONDOWN:
        colll.append(x)
        filll.append(y)
        counter = counter + 1

    if event == cv2.EVENT_RBUTTONDOWN:
        colll.pop(len(colll) - 1)
        filll.pop(len(filll) - 1)
        counter = counter - 1
# ----------------------------------------------------------------------------------------
def CrearMascara(file):
    """
    Create a custom mask for a given station from a video

    Inputs:
    	- file: String, with file name.

    Returns:
    	- Nothing returns. But save the obtained mask in the directory:
    	'Mascaras/Mascara_Station_' + str(int(f)) + '.png'
    """
    global filll
    global colll
    global counter
    filll = []
    colll=[]
    counter = -1

    f = numero_estacion(file)
    print('Creando máscara para la estación ', f, '...')

    video = VideoFileClip(file)  # Cargo el video
    frames = []
    for i in video.iter_frames():  # Obtengo los frames
        frames.append(i[:, :, 0])
    framesarray = numpy.asarray(frames)
    i = numpy.random.randint(0, len(frames))  # tomo frame aleatorio

    frame = framesarray[i]
    _, cantidad_filas, cantidad_columnas = framesarray.shape

    while True:
        k = cv2.waitKey(100)
        if k == 27:
            cv2.destroyAllWindows()
            break
        if counter != -1:
            frame = cv2.circle(frame, (colll[counter], filll[counter]), 2, (255, 0, 0), cv2.FILLED)
            if (counter > 0):
                frame = cv2.line(frame, (colll[counter - 1], filll[counter - 1]),
                                 (colll[counter], filll[counter]), (255, 0, 0))
        cv2.imshow('imagen', frame)
        cv2.setMouseCallback('imagen', MousePoints)
        cv2.waitKey(1)
    if (counter < 3):
        print('Ingrese una cantidad de puntos superior a 2...')
    else:
        mask = numpy.zeros((cantidad_filas, cantidad_columnas))
        r = numpy.asarray(filll)
        c = numpy.asarray(colll)
        rr, cc = skimage.draw.polygon(r, c, (cantidad_filas, cantidad_columnas))
        mask[rr, cc] = 1

        mpl.imsave('./Mascaras/Mascara_Station_' + str(int(f)) + '.png', mask,cmap=cm.gray)

    return


# ------------------------------------------------------------------------
def center_mass(frame, i_obj, j_obj, ventana):
    """
    Determine the center of mass of the event.

    Inputs:
    	- Frame: A 640x480 numpy array with one image of the video.
    	- i_obj: Integer with estimated event horizontal position.
    	- j_obj: Integer with estimated event vertical position.
    	- ventana: Integer with size of the window in which the center of mass is calculated.
    	A value of 5-6 pixels is recommended.

    Returns:
    	- x_c: Horizontal coordinate of the center of mass of the object.
    	- y_c: Vertical coordinate of the center of mass of the object.
    """
    ventana_i = ventana
    ventana_j = ventana
    Ancho, Alto = frame.shape
    while (i_obj - ventana < 0) or (i_obj + ventana > Ancho):
        ventana_i -= 1
    while (j_obj - ventana < 0) or (j_obj + ventana > Alto):
        ventana_j -= 1

    int_jvar = 0
    int_tot = 0
    int_xvar = 0
    int_ivar = 0
    int_yvar = 0
    for i in range(int(i_obj) - ventana_i, int(i_obj) + ventana_i + 1):
        for j in range(int(j_obj) - ventana_j, int(j_obj) + ventana_j + 1):
            int_jvar = int_jvar + frame[i, j]
        int_tot = int_jvar + int_tot
        int_xvar = int_xvar + i * int_jvar
        int_jvar = 0

    y_c = int_xvar / int_tot

    for j in range(int(j_obj) - ventana_j, int(j_obj) + ventana_j + 1):
        for i in range(int(i_obj) - ventana_i, int(i_obj) + ventana + 1):
            int_ivar = int_ivar + frame[i, j]
        int_yvar = int_yvar + j * int_ivar
        int_ivar = 0

    x_c = int_yvar / int_tot

    return x_c, y_c


# ------------------------------------------------------------


# -------------------------------------------------------------------------------

def CurvaDeLuz(Frame, Fondo, CM_f, CM_c, Size_Win):
    """
	Calculates the light intensity of the event with respect to the background.

	Inputs:
		- Frame: A 640x480 numpy array with one image of the video.
		- Fondo: A 640x480 numpy array with the background image of the video.
		- CM_f: Integer with vertical coordinate of the center of mass of the object.
		- CM_c: Integer with horizontal coordinate of the center of mass of the object.
		- Size_Win: Integer with size of the window in which the intensity is calculated.

	Returns:
		- LC: Integer with the value of light intensity of the event.
	"""

    lc = numpy.sum(
        Frame[0][round(CM_f) - Size_Win:round(CM_f) + Size_Win, round(CM_c) - Size_Win:round(CM_c) + Size_Win])
    fon = numpy.sum(Fondo[round(CM_f) - Size_Win:round(CM_f) + Size_Win, round(CM_c) - Size_Win:round(CM_c) + Size_Win])

    LC = lc - fon

    return LC


# ---------------------------------------''''----------------------------------------


def FWHM(Frame, CM_f, CM_c):
    """
    Calculates the FWHM - Full Width at Half Maximum

    Inputs:
    	- Frame: A 640x480 numpy array with one image of the video.
    	- CM_f: Integer with vertical coordinate of the center of mass of the object.
    	- CM_c: Integer with horizontal coordinate of the center of mass of the object.

    Returns:
    	- fwhm: Integer with the value of FWHM.
    	- fwhm_spix: Integer with the value of FWHM at the subpixel level.

    """
    '''
    Calcula el FWHM - Full Width at Half Maximum
    '''
    _, Ancho, Alto = Frame.shape
    w = 1
    Borde = 300  # Valor muy grande, importa que sea mayor al máximo de intensidad del objeto.
    Dif = 100  # Valor muy grande,importa que sea mayor a 2.
    Aux = Frame
    Int = Frame[0][round(CM_f), round(CM_c)]
    Entro = False

    while (Borde >= Int / 2) and (Dif > 0.01):
        # Se mantiene dentro del rango del video [0,Ancho-1]x[0,Alto-1]
        if (round(CM_f) - w < 0) or (round(CM_f) + w >= Ancho) or (round(CM_c) - w < 0) or (round(CM_c) + w >= Alto):
            return 255, 255

        else:
            # print('Dentro del while','Borde','Int','Int/2','Dif', Borde,Int,Int/2,Dif)
            Aux1 = Aux[0][round(CM_f) - w: round(CM_f) + w, round(CM_c) - w]
            + Aux[0][round(CM_f) - w: round(CM_f) + w, round(CM_c) + w]
            + Aux[0][round(CM_f) - w, round(CM_c) - w: round(CM_c) + w]
            + Aux[0][round(CM_f) + w, round(CM_c) - w: round(CM_c) + w]

            Dif = numpy.abs(Borde - numpy.mean(Aux1))
            Anterior = Borde
            Borde = numpy.mean(Aux1)
            # Si Borde incrementa por arriba del 20% del caso anterior
            # Entonces salgo del while y Entro asigno en verdadero
            # Para cambiar la forma en la que se define la recta.'
            if (Borde > 1.2 * Anterior):
                Dif = 0
                Entro = True

            w = w + 1

            # print('Al salir del while','Borde','Dif','Anterior',Borde,Dif,Anterior)
    w = w - 1
    # Se determina la recta que pasa entre los valores de fwhm(w) y fwhm(w+1)
    # Y se interpola en el valor Int/2

    if Entro:
        a = Anterior - Borde
        # Al estimar la pendiente es cero entonces devuelvo en ambos casos el fwhm entero
        if a == 0:
            return w, w
    else:
        a = Borde - Anterior
        # Al estimar la pendiente es cero entonces devuelvo en ambos casos el fwhm entero
        if a == 0:
            return w, w

    b = Anterior - a * w

    fwhm_spix = (Int / 2 - b) / a
    fwhm = w

    return fwhm, fwhm_spix


# ------------------------------------------------------------------------------

def xy_to_quadratic(xy):
    '''
    convierte los pares de coordenadas (x,y)
    en variables auxiliares w=(1,x,y,x^2,xy,y^2)

    la entrada consiste en una tupla con dos elementos:

    la primera es una lista con los valores de y (los indices de _fila_)
    la segunda es la lista correspondiente con los valores de x (los índices de _columna_)

    vienen así porque ese es el formato que devuelve la función numpy.nonzero,
    donde la primera lista tiene los índices de las filas (coordenada y) y el segundo
    el de las columnas (coordenada x), es decir, la convención de índices de matrices
    '''
    y = xy[0]  # está bien así: y va primero!
    x = xy[1]  # está bien así: x es la segunda lista

    n = len(x)
    m = 6
    X_poly = numpy.ones((n, m))
    X_poly[:, 1] = x
    X_poly[:, 2] = y
    X_poly[:, 3] = x ** 2
    X_poly[:, 4] = x * y
    X_poly[:, 5] = y ** 2
    return X_poly


# ------------------------------------------------------------------------------
def zoom_on_frame(frame, x, y, margin=20):
    """
    Get a zoomed box from a frame (A 640x480 numpy array)

    Inputs:
    	- frame: A 640x480 numpy array with one image of the video.
    	- x: Integer with vertical coordinate of interest.
    	- y: Integer with horizontal coordinate of interest.
    	- margin: Integer with size of the window. A value of 20 pixels is by default.

    Returns:
    	- frame[ymin:ymax, xmin:xmax]: A margin x margin numpy array.

    """
    h, w = frame.shape
    xmin, xmax = int(x - margin), min(int(x + margin), w - 1)
    ymin, ymax = int(y - margin), min(int(y + margin), h - 1)
    return frame[ymin:ymax, xmin:xmax]


# ------------------------------------------------------------------------------

def reconstruir_brillo(img, sat):
    '''
    reconstruye puntos saturados en la imagen 'img'
    usando un modelo cuadrático.
    el modelo se entrena en los puntos circundantes que están por encima
    de cierto umbral de intensidad
    '''
    M, N = img.shape
    #
    # usamos solamente los pixeles que no estan saturados
    # y que están por encima de cierto umbral
    #
    xy_sat = numpy.nonzero(sat)
    xy_train = numpy.nonzero(numpy.logical_and(numpy.greater(img, 140), numpy.logical_not(sat)))
    z_train = img[xy_train]

    X_sat = xy_to_quadratic(xy_sat)
    X_train = xy_to_quadratic(xy_train)
    #
    # ajustamos modelo cuadrático
    #
    res = numpy.linalg.lstsq(X_train, z_train)
    a = res[0]
    #
    # evaluamos en puntos desconocidos
    #
    z_pred = numpy.dot(X_sat, a)
    rec = numpy.copy(img)
    rec[xy_sat] = z_pred
    return rec


# ------------------------------------------------------------------------------


def filtro_extra(X_bol, Y_bol, pertenecen2):
    '''
    Eliminates outliers that fits in the model but are caused by other events or noise. It's a second filter after RANSAC


    :param X_bol: x axis coordinates of the event
    :param Y_bol: y axis coordinates of the event
    :param pertenecen2: boolean list of the ransac model
    :return: modified boolean list of the ransac model
    '''


    X_ord = sorted(X_bol)  # se ordenan las posiciones de menor a mayor
    Y_ord = sorted(Y_bol)  # se ordenan las posiciones de menor a mayor
    if len(X_ord) / 2 == 0:
        n = len(X_ord) / 2
        n = int(n)
    else:
        n = (len(X_ord) - 1) / 2
        n = int(n)

    X_Q1 = numpy.median(X_ord[:n])
    X_Q3 = numpy.median(X_ord[n:])
    X_IQR = X_Q3 - X_Q1
    X_High_outlier = X_Q3 + 1.5 * X_IQR
    X_Low_outlier = X_Q1 - 1.5 * X_IQR

    Y_Q1 = numpy.median(Y_ord[:n])
    Y_Q3 = numpy.median(Y_ord[n:])
    Y_IQR = Y_Q3 - Y_Q1
    Y_High_outlier = Y_Q3 + 1.5 * Y_IQR
    Y_Low_outlier = Y_Q1 - 1.5 * Y_IQR

    for i in range(len(pertenecen2)):
        if X_bol[i] < X_Low_outlier - 1 or X_bol[i] > X_High_outlier + 1 or Y_bol[i] < Y_Low_outlier - 1 or Y_bol[
            i] > Y_High_outlier + 1:
            pertenecen2[i] = False

    XBol_aux = X_bol[pertenecen2]
    YBol_aux = Y_bol[pertenecen2]

    XBol_aux = sorted(XBol_aux)
    YBol_aux = sorted(YBol_aux)

    if len(XBol_aux) / 2 == 0:
        n = len(XBol_aux) / 2
        n = int(n)
    else:
        n = (len(XBol_aux) - 1) / 2
        n = int(n)

    X_Q1 = numpy.median(XBol_aux[:n])
    X_Q3 = numpy.median(XBol_aux[n:])
    X_IQR = X_Q3 - X_Q1
    X_High_outlier = X_Q3 + 1.5 * X_IQR
    X_Low_outlier = X_Q1 - 1.5 * X_IQR

    Y_Q1 = numpy.median(YBol_aux[:n])
    Y_Q3 = numpy.median(YBol_aux[n:])
    Y_IQR = Y_Q3 - Y_Q1
    Y_High_outlier = Y_Q3 + 1.5 * Y_IQR
    Y_Low_outlier = Y_Q1 - 1.5 * Y_IQR

    for i in range(len(pertenecen2)):
        if X_bol[i] < X_Low_outlier - 1 or X_bol[i] > X_High_outlier + 1 or Y_bol[i] < Y_Low_outlier - 1 or Y_bol[
            i] > Y_High_outlier + 1:
            pertenecen2[i] = False

    return pertenecen2


# -------------------------------------------------------------------------------

def dispersion(x, y):
    """
    Calculates the dispersion between the event positions

    Inputs:
    	- x: Integer with vertical coordinate of interest.
    	- y: Integer with horizontal coordinate of interest.

    Returns:
    	- disper: Integer with the sum of each of the distances of the event
    	positions with respect to their midpoint, normalized.

    """
    x_mean = numpy.mean(x)
    y_mean = numpy.mean(y)
    c = [x_mean, y_mean]
    disper = 0
    for i in range(len(x)):
        a = x[i]
        b = y[i]
        disper = disper + 1 / len(x) * numpy.sqrt((a - c[0]) ** 2 + (b - c[1]) ** 2)

    return disper


def Clasificar(video):
    """
    	Function that gets some of the characteristics of a video event.

    	Inputs:
    		- video: String, with the location and name of the file to execute.

    	Returns:
    		- velocidad_media: Integer, with the average speed of the event.
    		- mean_intensidad: Integer, with the average light intensity of the event.
    		- N: Integer, with the value of the size of the numpy array that contains the frames of the video.
    		- total_dispersion: Integer, with the value of the average dispersion of the event positions in
    		the frames.
    		- max_vel: Integer, with the maximum speed of the event.
    		- max_int: Integer, with the maximum light intensity of the event.
    		- min_vel: Integer, with the minimum speed of the event.
    		- min_int: Integer, with the minimum light intensity of the event.
    		- var_int: Integer, with the variance speed of the event.
    		- var_vel: Integer, with the variance light intensity of the event.
    		- mean_fwhm: Integer, with the average FWHM of the event.
    		- var_fwhm: Integer, with the variance FWHM of the event.
    		- V_feat_i_1: Intenger, with the softness of light intensity.
    		- V_feat_i_2: Intenger, with the softness of light intensity normalized.
    		- V_feat_v_1: Intenger, with the softness of speed of the event.
    		- V_feat_v_2: Intenger, with the softness of speed of the event normalized.
    		- V_feat_fwhm_1: Intenger, with the softness of FWHM of the event.
    		- V_feat_fwhm_2: Intenger, with the softness of FWHM of the event normalized.
    		- loss: Integer, with the mean square error between the trajectory it makes and the model of a
    		circumference.
    		- d: Integer, with the value of the distance traveled.
    	"""
    ruta_entrada, t, x, y, i, N, total_dispersion, FWHM, loss, d = tracking(video)
    if numpy.isnan(t[0]) or numpy.isnan(i[0]):
        return ruta_entrada, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        vel = velocidad(t, x, y)
        if bool(vel) and bool(i) and bool(FWHM):
            velocidad_media = numpy.mean(numpy.array(vel))
            var_vel = numpy.var(numpy.array(vel))
            max_vel = numpy.max(numpy.array(vel))
            min_vel = numpy.min(numpy.array(vel))
            max_int = numpy.max(numpy.array(i))
            min_int = numpy.min(numpy.array(i))
            var_int = numpy.var(numpy.array(i))
            mean_intensidad = numpy.mean(i)
            mean_fwhm = numpy.mean(FWHM)
            var_fwhm = numpy.var(FWHM)
            V_feat_i_1, V_feat_i_2 = V_feature(vel)
            V_feat_v_1, V_feat_v_2 = V_feature(i)
            V_feat_fwhm_1, V_feat_fwhm_2 = V_feature(FWHM)
            return ruta_entrada, velocidad_media, mean_intensidad, N, total_dispersion, max_vel, max_int, min_vel, min_int, var_int, var_vel, mean_fwhm, var_fwhm, V_feat_i_1, V_feat_i_2, V_feat_v_1, V_feat_v_2, V_feat_fwhm_1, V_feat_fwhm_2, loss, d
        else:
            return ruta_entrada, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
