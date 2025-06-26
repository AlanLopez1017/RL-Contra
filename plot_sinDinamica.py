import numpy as np

# Cargar el archivo .npy
datos = np.load('rewards_contra_new_dinamica_last.npy')  # Reemplaza con tu ruta real


import matplotlib.pyplot as plt

plt.plot(datos)
plt.title("Gráfica de datos")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
