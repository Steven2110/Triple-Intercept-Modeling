import numpy as np
import logging

from constant import SATELLITE1, SATELLITE2, SATELLITE3, SATELLITE4, Coordinate

# Set up logging configuration:
logging.basicConfig(
    filename="newton_log.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calculate_distance(satellite1, satellite2):
    return np.sqrt(
        (satellite1.x - satellite2.x)**2 +
        (satellite1.y - satellite2.y)**2 +
        (satellite1.z - satellite2.z)**2
    )
    
def calculate_derivative(satellite1, satellite2, rho):
    drho_i_dx = (satellite1.x - satellite2.x) / rho
    drho_i_dy = (satellite1.y - satellite2.y) / rho
    drho_i_dz = (satellite1.z - satellite2.z) / rho
    
    return drho_i_dx, drho_i_dy, drho_i_dz

def main(max_iter=10_000, tol=1e-10):
    # Observations distances
    rho_1_o = calculate_distance(SATELLITE4, SATELLITE1)
    rho_2_o = calculate_distance(SATELLITE4, SATELLITE2)
    rho_3_o = calculate_distance(SATELLITE4, SATELLITE3)
    logging.info("Observed distances (ρᵢ):")
    logging.info(f"ρ₁ = {rho_1_o}")
    logging.info(f"ρ₂ = {rho_2_o}")
    logging.info(f"ρ₃ = {rho_3_o}")

    # Calculated distances
    # Initial guess
    GUESS_SATELLITE = Coordinate(
        x=0.0,
        y=0.0,
        z=0.0
    )

    eps = 1e-12

    for i in range(max_iter):
        logging.info(f"Iteration: {i + 1}")
        # 1. Calculate calculated rho (ρᵢ) with current guess
        rho1 = calculate_distance(GUESS_SATELLITE, SATELLITE1)
        rho2 = calculate_distance(GUESS_SATELLITE, SATELLITE2)
        rho3 = calculate_distance(GUESS_SATELLITE, SATELLITE3)

        # 2. Calculate delta rho (Δρᵢ) - ρᵢ O - ρᵢ C.
        delta_rho1 = rho_1_o - rho1
        delta_rho2 = rho_2_o - rho2
        delta_rho3 = rho_3_o - rho3

        # Вывод результатов:
        logging.info("Computed distances from the current guess (ρᵢ):")
        logging.info(f"ρ₁ = {rho1}")
        logging.info(f"ρ₂ = {rho2}")
        logging.info(f"ρ₃ = {rho3}")

        logging.info("Δρᵢ (разность между смоделированными \
                     и вычисленными расстояниями):")
        logging.info(f"Δρ₁ = {delta_rho1}")
        logging.info(f"Δρ₂ = {delta_rho2}")
        logging.info(f"Δρ₃ = {delta_rho3}")

        # Protect against division by zero (if any computed rho_i is too small)
        rho1 = rho1 if rho1 > eps else eps
        rho2 = rho2 if rho2 > eps else eps
        rho3 = rho3 if rho3 > eps else eps

        # Compute partial derivatives for Satellite 1
        drho_1_dx, drho_1_dy, drho_1_dz = calculate_derivative(GUESS_SATELLITE, SATELLITE1, rho1)
        # Compute partial derivatives for Satellite 2
        drho_2_dx, drho_2_dy, drho_2_dz = calculate_derivative(GUESS_SATELLITE, SATELLITE2, rho2)
        # Compute partial derivatives for Satellite 3
        drho_3_dx, drho_3_dy, drho_3_dz = calculate_derivative(GUESS_SATELLITE, SATELLITE3, rho3)

        # Form the matrix A
        A = np.array([
            [drho_1_dx, drho_1_dy, drho_1_dz],
            [drho_2_dx, drho_2_dy, drho_2_dz],
            [drho_3_dx, drho_3_dy, drho_3_dz]
        ])

        logging.info("Matrix A:")
        logging.info(A)

        # Формируем вектор Δρ
        b = np.array([delta_rho1, delta_rho2, delta_rho3])
        logging.info(f"Vector Δρ (b): {b}")

        # Решаем систему A * Δx = b для вектора поправок Δx = [Δx, Δy, Δz]^T
        delta_x = np.linalg.inv(A) @ b

        # Извлекаем поправки
        Delta_x = delta_x[0]
        Delta_y = delta_x[1]
        Delta_z = delta_x[2]

        logging.info(f"Corrections (Δx, Δy, Δz): {Delta_x}, {Delta_y}, {Delta_z}")
        logging.info("Current Guess (x, y, z): {GUESS_SATELLITE.x}, {GUESS_SATELLITE.y}, {GUESS_SATELLITE.z}")
        
        previous_guess = np.array([GUESS_SATELLITE.x, GUESS_SATELLITE.y, GUESS_SATELLITE.z])
        
        # Обновляем текущее приближение
        GUESS_SATELLITE.x += Delta_x
        GUESS_SATELLITE.y += Delta_y
        GUESS_SATELLITE.z += Delta_z
        
        current_guess = np.array([GUESS_SATELLITE.x, GUESS_SATELLITE.y, GUESS_SATELLITE.z])
        
        logging.info("Updated guess (x, y, z): {GUESS_SATELLITE.x}, {GUESS_SATELLITE.y}, {GUESS_SATELLITE.z}")
        
        if np.linalg.norm(current_guess - previous_guess) < tol:
            print("Converged to a solution.")
            logging.info(f"Converged to a solution on iteration {i + 1}.")
            logging.info(f"Final estimated coordinates (x, y, z): {GUESS_SATELLITE.x}, {GUESS_SATELLITE.y}, {GUESS_SATELLITE.z}")
            return GUESS_SATELLITE, i+1
        
estimated_coordinates, iteration = main()
print(f"Iteration: {iteration}")
print("Estimated coordinates (x, y, z):", estimated_coordinates.x, estimated_coordinates.y, estimated_coordinates.z)
print("Original coordinates (x, y, z):", SATELLITE4.x, SATELLITE4.y, SATELLITE4.z)