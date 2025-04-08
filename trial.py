import numpy as np
from constant import SATELLITE1, SATELLITE2, SATELLITE3

x = 42122.835
y = 0.0
z = 0.0

# SYSTEM LINEAR

rho_1_o = np.sqrt(
    (x - SATELLITE1.x)**2 +
    (y - SATELLITE1.y)**2 +
    (z - SATELLITE1.z)**2
)

rho_2_o = np.sqrt(
    (x - SATELLITE2.x)**2 +
    (y - SATELLITE2.y)**2 +
    (z - SATELLITE2.z)**2
)

rho_3_o = np.sqrt(
    (x - SATELLITE3.x)**2 +
    (y - SATELLITE3.y)**2 +
    (z - SATELLITE3.z)**2
)

print("Distance to Satellite 1 =", rho_1_o)
print("Distance to Satellite 2 =", rho_2_o)
print("Distance to Satellite 3 =", rho_3_o)


x_guess = 0.0
y_guess = 0.0
z_guess = 0.0

# 1. Вычисляем вычисленные расстояния (ρᵢ) от текущего приближения (x, y, z) до каждой спутниковой точки.
rho1 = np.sqrt((x_guess - SATELLITE1.x)**2 + (y_guess - SATELLITE1.y)**2 + (z_guess - SATELLITE1.z)**2)
rho2 = np.sqrt((x_guess - SATELLITE2.x)**2 + (y_guess - SATELLITE2.y)**2 + (z_guess - SATELLITE2.z)**2)
rho3 = np.sqrt((x_guess - SATELLITE3.x)**2 + (y_guess - SATELLITE3.y)**2 + (z_guess - SATELLITE3.z)**2)

# 2. Вычисляем разность (Δρᵢ) между смоделированными (истинными) расстояниями (ρᵢ⁰) 
#    и вычисленными расстояниями ρᵢ при текущем приближении:
delta_rho1 = rho_1_o - rho1
delta_rho2 = rho_2_o - rho2
delta_rho3 = rho_3_o - rho3

# Вывод результатов:
print("Computed distances from the current guess (ρᵢ):")
print("ρ₁ =", rho1)
print("ρ₂ =", rho2)
print("ρ₃ =", rho3)

print("\nΔρᵢ (разность между смоделированными и вычисленными расстояниями):")
print("Δρ₁ =", delta_rho1)
print("Δρ₂ =", delta_rho2)
print("Δρ₃ =", delta_rho3)

eps = 1e-12

# Protect against division by zero (if any computed rho_i is too small)
rho1_safe = rho1 if rho1 > eps else eps
rho2_safe = rho2 if rho2 > eps else eps
rho3_safe = rho3 if rho3 > eps else eps

# Compute partial derivatives for Satellite 1
drho1_dx = (x_guess - SATELLITE1.x) / rho1_safe
drho1_dy = (y_guess - SATELLITE1.y) / rho1_safe
drho1_dz = (z_guess - SATELLITE1.z) / rho1_safe

# Compute partial derivatives for Satellite 2
drho2_dx = (x_guess - SATELLITE2.x) / rho2_safe
drho2_dy = (y_guess - SATELLITE2.y) / rho2_safe
drho2_dz = (z_guess - SATELLITE2.z) / rho2_safe

# Compute partial derivatives for Satellite 3
drho3_dx = (x_guess - SATELLITE3.x) / rho3_safe
drho3_dy = (y_guess - SATELLITE3.y) / rho3_safe
drho3_dz = (z_guess - SATELLITE3.z) / rho3_safe

# Form the Jacobian matrix A = F'(x)
A = np.array([
    [drho1_dx, drho1_dy, drho1_dz],
    [drho2_dx, drho2_dy, drho2_dz],
    [drho3_dx, drho3_dy, drho3_dz]
])

print("Jacobian Matrix A = F'(x):")
print(A)

# Формируем вектор Δρ
b = np.array([delta_rho1, delta_rho2, delta_rho3])
print("Vector Δρ (b):", b)

# Решаем систему A * Δx = b для вектора поправок Δx = [Δx, Δy, Δz]^T
delta_x = np.linalg.solve(A, b)

# Извлекаем поправки
Delta_x = delta_x[0]
Delta_y = delta_x[1]
Delta_z = delta_x[2]

print("Corrections (Δx, Δy, Δz):", Delta_x, Delta_y, Delta_z)
print("Guess (x, y, z):", x_guess, y_guess, z_guess)
# Обновляем текущее приближение
x_guess += Delta_x
y_guess += Delta_y
z_guess += Delta_z
print("Updated guess (x, y, z):", x_guess, y_guess, z_guess)


# 2nd iteration
# 1. Вычисляем вычисленные расстояния (ρᵢ) от текущего приближения (x, y, z) до каждой спутниковой точки.
rho1 = np.sqrt((x_guess - SATELLITE1.x)**2 + (y_guess - SATELLITE1.y)**2 + (z_guess - SATELLITE1.z)**2)
rho2 = np.sqrt((x_guess - SATELLITE2.x)**2 + (y_guess - SATELLITE2.y)**2 + (z_guess - SATELLITE2.z)**2)
rho3 = np.sqrt((x_guess - SATELLITE3.x)**2 + (y_guess - SATELLITE3.y)**2 + (z_guess - SATELLITE3.z)**2)

# 2. Вычисляем разность (Δρᵢ) между смоделированными (истинными) расстояниями (ρᵢ⁰) 
#    и вычисленными расстояниями ρᵢ при текущем приближении:
delta_rho1 = rho_1_o - rho1
delta_rho2 = rho_2_o - rho2
delta_rho3 = rho_3_o - rho3

# Вывод результатов:
print("Computed distances from the current guess (ρᵢ):")
print("ρ₁ =", rho1)
print("ρ₂ =", rho2)
print("ρ₃ =", rho3)

print("\nΔρᵢ (разность между смоделированными и вычисленными расстояниями):")
print("Δρ₁ =", delta_rho1)
print("Δρ₂ =", delta_rho2)
print("Δρ₃ =", delta_rho3)

eps = 1e-12

# Protect against division by zero (if any computed rho_i is too small)
rho1_safe = rho1 if rho1 > eps else eps
rho2_safe = rho2 if rho2 > eps else eps
rho3_safe = rho3 if rho3 > eps else eps

# Compute partial derivatives for Satellite 1
drho1_dx = (x_guess - SATELLITE1.x) / rho1_safe
drho1_dy = (y_guess - SATELLITE1.y) / rho1_safe
drho1_dz = (z_guess - SATELLITE1.z) / rho1_safe

# Compute partial derivatives for Satellite 2
drho2_dx = (x_guess - SATELLITE2.x) / rho2_safe
drho2_dy = (y_guess - SATELLITE2.y) / rho2_safe
drho2_dz = (z_guess - SATELLITE2.z) / rho2_safe

# Compute partial derivatives for Satellite 3
drho3_dx = (x_guess - SATELLITE3.x) / rho3_safe
drho3_dy = (y_guess - SATELLITE3.y) / rho3_safe
drho3_dz = (z_guess - SATELLITE3.z) / rho3_safe

# Form the Jacobian matrix A = F'(x)
A = np.array([
    [drho1_dx, drho1_dy, drho1_dz],
    [drho2_dx, drho2_dy, drho2_dz],
    [drho3_dx, drho3_dy, drho3_dz]
])

print("Jacobian Matrix A = F'(x):")
print(A)

# Формируем вектор Δρ
b = np.array([delta_rho1, delta_rho2, delta_rho3])
print("Vector Δρ (b):", b)

# Решаем систему A * Δx = b для вектора поправок Δx = [Δx, Δy, Δz]^T
delta_x = np.linalg.solve(A, b)

# Извлекаем поправки
Delta_x = delta_x[0]
Delta_y = delta_x[1]
Delta_z = delta_x[2]

print("Corrections (Δx, Δy, Δz):", Delta_x, Delta_y, Delta_z)
print("Guess (x, y, z):", x_guess, y_guess, z_guess)
# Обновляем текущее приближение
x_guess += Delta_x
y_guess += Delta_y
z_guess += Delta_z
print("Updated guess (x, y, z):", x_guess, y_guess, z_guess)

# 3rd iteration
# 1. Вычисляем вычисленные расстояния (ρᵢ) от текущего приближения (x, y, z) до каждой спутниковой точки.
rho1 = np.sqrt((x_guess - SATELLITE1.x)**2 + (y_guess - SATELLITE1.y)**2 + (z_guess - SATELLITE1.z)**2)
rho2 = np.sqrt((x_guess - SATELLITE2.x)**2 + (y_guess - SATELLITE2.y)**2 + (z_guess - SATELLITE2.z)**2)
rho3 = np.sqrt((x_guess - SATELLITE3.x)**2 + (y_guess - SATELLITE3.y)**2 + (z_guess - SATELLITE3.z)**2)

# 2. Вычисляем разность (Δρᵢ) между смоделированными (истинными) расстояниями (ρᵢ⁰) 
#    и вычисленными расстояниями ρᵢ при текущем приближении:
delta_rho1 = rho_1_o - rho1
delta_rho2 = rho_2_o - rho2
delta_rho3 = rho_3_o - rho3

# Вывод результатов:
print("Computed distances from the current guess (ρᵢ):")
print("ρ₁ =", rho1)
print("ρ₂ =", rho2)
print("ρ₃ =", rho3)

print("\nΔρᵢ (разность между смоделированными и вычисленными расстояниями):")
print("Δρ₁ =", delta_rho1)
print("Δρ₂ =", delta_rho2)
print("Δρ₃ =", delta_rho3)

eps = 1e-12

# Protect against division by zero (if any computed rho_i is too small)
rho1_safe = rho1 if rho1 > eps else eps
rho2_safe = rho2 if rho2 > eps else eps
rho3_safe = rho3 if rho3 > eps else eps

# Compute partial derivatives for Satellite 1
drho1_dx = (x_guess - SATELLITE1.x) / rho1_safe
drho1_dy = (y_guess - SATELLITE1.y) / rho1_safe
drho1_dz = (z_guess - SATELLITE1.z) / rho1_safe

# Compute partial derivatives for Satellite 2
drho2_dx = (x_guess - SATELLITE2.x) / rho2_safe
drho2_dy = (y_guess - SATELLITE2.y) / rho2_safe
drho2_dz = (z_guess - SATELLITE2.z) / rho2_safe

# Compute partial derivatives for Satellite 3
drho3_dx = (x_guess - SATELLITE3.x) / rho3_safe
drho3_dy = (y_guess - SATELLITE3.y) / rho3_safe
drho3_dz = (z_guess - SATELLITE3.z) / rho3_safe

# Form the Jacobian matrix A = F'(x)
A = np.array([
    [drho1_dx, drho1_dy, drho1_dz],
    [drho2_dx, drho2_dy, drho2_dz],
    [drho3_dx, drho3_dy, drho3_dz]
])

print("Jacobian Matrix A = F'(x):")
print(A)

# Формируем вектор Δρ
b = np.array([delta_rho1, delta_rho2, delta_rho3])
print("Vector Δρ (b):", b)

# Решаем систему A * Δx = b для вектора поправок Δx = [Δx, Δy, Δz]^T
delta_x = np.linalg.solve(A, b)

# Извлекаем поправки
Delta_x = delta_x[0]
Delta_y = delta_x[1]
Delta_z = delta_x[2]

print("Corrections (Δx, Δy, Δz):", Delta_x, Delta_y, Delta_z)
print("Guess (x, y, z):", x_guess, y_guess, z_guess)
# Обновляем текущее приближение
x_guess += Delta_x
y_guess += Delta_y
z_guess += Delta_z
print("Updated guess (x, y, z):", x_guess, y_guess, z_guess)

# 4th iteration
# 1. Вычисляем вычисленные расстояния (ρᵢ) от текущего приближения (x, y, z) до каждой спутниковой точки.
rho1 = np.sqrt((x_guess - SATELLITE1.x)**2 + (y_guess - SATELLITE1.y)**2 + (z_guess - SATELLITE1.z)**2)
rho2 = np.sqrt((x_guess - SATELLITE2.x)**2 + (y_guess - SATELLITE2.y)**2 + (z_guess - SATELLITE2.z)**2)
rho3 = np.sqrt((x_guess - SATELLITE3.x)**2 + (y_guess - SATELLITE3.y)**2 + (z_guess - SATELLITE3.z)**2)

# 2. Вычисляем разность (Δρᵢ) между смоделированными (истинными) расстояниями (ρᵢ⁰) 
#    и вычисленными расстояниями ρᵢ при текущем приближении:
delta_rho1 = rho_1_o - rho1
delta_rho2 = rho_2_o - rho2
delta_rho3 = rho_3_o - rho3

# Вывод результатов:
print("Computed distances from the current guess (ρᵢ):")
print("ρ₁ =", rho1)
print("ρ₂ =", rho2)
print("ρ₃ =", rho3)

print("\nΔρᵢ (разность между смоделированными и вычисленными расстояниями):")
print("Δρ₁ =", delta_rho1)
print("Δρ₂ =", delta_rho2)
print("Δρ₃ =", delta_rho3)

eps = 1e-12

# Protect against division by zero (if any computed rho_i is too small)
rho1_safe = rho1 if rho1 > eps else eps
rho2_safe = rho2 if rho2 > eps else eps
rho3_safe = rho3 if rho3 > eps else eps

# Compute partial derivatives for Satellite 1
drho1_dx = (x_guess - SATELLITE1.x) / rho1_safe
drho1_dy = (y_guess - SATELLITE1.y) / rho1_safe
drho1_dz = (z_guess - SATELLITE1.z) / rho1_safe

# Compute partial derivatives for Satellite 2
drho2_dx = (x_guess - SATELLITE2.x) / rho2_safe
drho2_dy = (y_guess - SATELLITE2.y) / rho2_safe
drho2_dz = (z_guess - SATELLITE2.z) / rho2_safe

# Compute partial derivatives for Satellite 3
drho3_dx = (x_guess - SATELLITE3.x) / rho3_safe
drho3_dy = (y_guess - SATELLITE3.y) / rho3_safe
drho3_dz = (z_guess - SATELLITE3.z) / rho3_safe

# Form the Jacobian matrix A = F'(x)
A = np.array([
    [drho1_dx, drho1_dy, drho1_dz],
    [drho2_dx, drho2_dy, drho2_dz],
    [drho3_dx, drho3_dy, drho3_dz]
])

print("Jacobian Matrix A = F'(x):")
print(A)

# Формируем вектор Δρ
b = np.array([delta_rho1, delta_rho2, delta_rho3])
print("Vector Δρ (b):", b)

# Решаем систему A * Δx = b для вектора поправок Δx = [Δx, Δy, Δz]^T
delta_x = np.linalg.solve(A, b)

# Извлекаем поправки
Delta_x = delta_x[0]
Delta_y = delta_x[1]
Delta_z = delta_x[2]

print("Corrections (Δx, Δy, Δz):", Delta_x, Delta_y, Delta_z)
print("Guess (x, y, z):", x_guess, y_guess, z_guess)
# Обновляем текущее приближение
x_guess += Delta_x
y_guess += Delta_y
z_guess += Delta_z
print("Updated guess (x, y, z):", x_guess, y_guess, z_guess)

