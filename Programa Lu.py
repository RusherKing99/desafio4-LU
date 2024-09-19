import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

# Definir la matriz A y los vectores B1 y B2
A = np.array([[10, 2, -1], [-3, -6, 2], [1, 1, 5]])
B1 = np.array([27, -61.5, -21.5])
B2 = np.array([12, 18, -6])

# Paso 1: Realizar la factorización LU
P, L, U = lu(A)

# Mostrar las matrices L, U y P
print("Matriz L (triangular inferior):")
print(L)
print("\nMatriz U (triangular superior):")
print(U)
print("\nMatriz P (de permutación):")
print(P)

# Paso 2: Resolver L * d = P * B1 (sustitución hacia adelante)
pb1 = np.dot(P, B1)
d1 = np.linalg.solve(L, pb1)

# Paso 3: Resolver U * x = d (sustitución hacia atrás)
x1 = np.linalg.solve(U, d1)

print("\nSolución con B1:")
print(x1)

# Parte c) Resolver el sistema con el vector alternativo B2
pb2 = np.dot(P, B2)
d2 = np.linalg.solve(L, pb2)
x2 = np.linalg.solve(U, d2)

print("\nSolución con B2:")
print(x2)

# Graficar las soluciones
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar el vector x obtenido con B1
ax.scatter(x1[0], x1[1], x1[2], color='r', s=100, label='Solución con B1')
ax.text(x1[0], x1[1], x1[2], ' B1', color='red')

# Graficar el vector x2 obtenido con B2
ax.scatter(x2[0], x2[1], x2[2], color='b', s=100, label='Solución con B2')
ax.text(x2[0], x2[1], x2[2], ' B2', color='blue')

# Configurar el gráfico
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Soluciones del sistema para B1 y B2')

ax.legend()
plt.show()
