# Snake-AI-Game

Tenemos que crear tres Módulos para este proyecto:

- El Entorno (el juego que construimos)
- El Modelo (Modelo de refuerzo para la predicción de movimientos)
- El Agente (Intermediario entre el Entorno y el Modelo)
  
![download](https://github.com/JordiPG05/Snake-AI-Game/assets/100807571/0a17a952-9e53-4870-8f6a-63dabf3c3400)

## Algoritmo

En el display disponemos de la serpiente y la fruta.

Primero debemos calcular el estado de la serpiente, para cada posición se calculara el estado de la serpiente determinado por un vector de 11 valores.

Una vez tengamos el vector, el agente pasara los datos al modelo, este modelo recompensara de forma que:
- Comer fruta: +10
- Juego terminado: -10
- Otros: +0

Se actualizara el valor de Q y se entrenara el modelo.

## El modelo
La idea base del modelo es esta:

![Model1](https://github.com/JordiPG05/Snake-AI-Game/assets/100807571/0d67c6bb-5afa-48c1-a0aa-509849557cb9)

Utilizaremos en este caso PyTorch pero se puede utilizar cualquier modelo de refuerzo por vectores (como TensorFlow).

Crearemos una RNN de una capa de entrada de tamaño 11, una capa densa y una capa de salida de 3 neuronas (recto, derecha, izquierda). *PUEDEN SER NECESARIAS MODIFICACIONES POSTERIORES*.
