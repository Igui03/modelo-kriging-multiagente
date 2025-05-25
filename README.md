# Búsqueda del Máximo Global de un Campo Escalar usando Kriging y Sistemas Multiagente

Este repositorio contiene el código y los recursos asociados a un estudio sobre la **optimización global de campos escalares** mediante técnicas de **kriging** y el uso de un **sistema multiagente** colaborativo. El objetivo es localizar el máximo global de un campo gaussiano en un dominio dado, utilizando información recolectada secuencialmente por agentes.



## Descripción del proyecto

El trabajo implementa un enfoque basado en:

- Modelado del campo como un proceso gaussiano.
- Estimación mediante kriging ordinario y universal.
- Actualización iterativa del modelo a partir de mediciones locales de agentes móviles.
- Estrategias de decisión descentralizadas y colaboración entre agentes.
- Aplicación a campos simulados con parámetros ajustables (varianza, correlación espacial, etc.).

Este código fue desarrollado como parte de un estudio académico y acompaña un artículo técnico sobre el tema.



## Estructura del repositorio

memoria_tfg # La memoria sobre el trabajo en el que se muestran los resultados de las simulaciones
cond_bayesiano.py # Contiene el programa necesario para realizar la gráfica del condicionamiento bayesiano de la memoria
modelo_kriging_universal # Simulación correspondiente a la prueba de convergencia del algoritmo (usar semillas s=12,20)
modelo_kriging_ordinario # Simulación correspondiente a la comparación del kriging ordinario y universal
modelo_expected_improvement # Simulación correspondiente a la comparación de las distintas funciones de adquisición planteadas en la memoria

## Requisitos

El código está escrito en Python 3 y requiere las siguientes bibliotecas:

- pykrige
- gstools
- numpy  
- scipy  
- matplotlib  

