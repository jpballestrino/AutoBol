# AutoBol

## Clasificador automático de videos para proyecto Bocosur

## Descripción

Autobol es un programa implementado en Python que tiene como objetivo detectar bólidos en los videos obtenidos por las estaciones con cámaras allsky localizadas a través del territorio Uruguayo. Es parte de un proyecto principal llamado BOCOSUR (http://bolidos.astronomia.edu.uy/).
Este programa tiene dos funciones principales, clasificar videos y reentrenar el modelo.

## Instalación

Este programa está 100% implementado en Python 3, software de uso libre y gratuito. 

### Dependencies

Los paquetes necesarios para correr AUTOBOL son: ```NumPy```, ```SciPy```, ```matplotlib```, ```xgboost```, ```moviepy```, ```cv2```, ```pandas```, ```scikit-image```, ```pyfiglet``` ```sklearn```

Estos se pueden instalar mediante los comandos:

pip install numpy scipy matplotlib moviepy pandas xgboost scikit-image pyfiglet sklearn

pip install opencv-python



## Usage
-o : indica la ruta para guardar la salida
-i : indica la ruta de entrada de los archivos
-mod: Elección del modelo a utilizar, válido solo para predict
-u : Umbral de decisión, válido sólo para predict
-t : test_size, válido solo para train
-mn : Nombre del modelo al guardarlo, valido solo para train



Function mask: Once the mask is finished press Esc button to exit the function.
