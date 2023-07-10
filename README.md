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
