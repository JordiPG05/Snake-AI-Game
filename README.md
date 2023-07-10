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

### Funcionamiento del modelo

Con esta primera idea en la cabeza, vamos a definir el funcionamiento del modelo.

Siguiendo la base de la ecuación de Bellman:

![quicklatex com-70991fb14d9c469b4f9a35fb6ad8cb9d_l3](https://github.com/JordiPG05/Snake-AI-Game/assets/100807571/157e2f43-6be6-45d4-8497-6a81427d1c29)

Adaptando esta ecuación, obtenemos que:

![image2](https://github.com/JordiPG05/Snake-AI-Game/assets/100807571/8647b239-952d-4e09-b967-e54060bfdc2b)

Crearemos tambien una variable que almacene el estado original, la acción y el estado posterior. De esta forma el modelo tendra una memoria de repetición y mejorara su rendimiento.

Estas dos operaciones entran en bucle hasta que finaliza el juego.

**LA CLAVE ES CREAR UN BUEN MODELO** ya que la precisión de la serpiente depende de ello.



