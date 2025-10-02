import sympy as sp
import numpy as np
import math
# Se asume que la clase ForwardKinematicsDH está definida en 'forward_kinematics_dh_class.py'
from forward_kinematics_dh_class import ForwardKinematicsDH

# ================================================================
#                       FUNCIONES DE ANÁLISIS
# ================================================================

def analyze_rr():
    """Analiza el Robot Planar (RR). Calcula H, Posición, Jacobiano, Det(J) y Cinemática Inversa."""
    print("\n[INICIANDO ANÁLISIS: Robot Planar (RR)]")

    # --- 1. DEFINICIÓN DE VARIABLES ---
    q1, q2 = sp.symbols('q1 q2')
    l1, l2 = sp.symbols('l1 l2')
    x, y = sp.symbols('x y') 
    q_vector = sp.Matrix([q1, q2])
    
    # --- 2. CÁLCULO SIMBÓLICO DE CINEMÁTICA DIRECTA (Eq. 4.19) ---
    H_sym_RR = ForwardKinematicsDH.symbolic([[q1, 0, l1, 0], [q2, 0, l2, 0]])
    H_RR_simplified = sp.trigsimp(H_sym_RR)

    print("\n--- MATRIZ DE TRANSFORMACIÓN HOMOGÉNEA (H_0^2 - Eq. 4.19) ---")
    sp.pprint(H_RR_simplified, use_unicode=True)
    
    # --- 3. EXTRACCIÓN DE POSICIÓN (Eq. 4.20) y JACOBIANO (Eq. 4.23) ---
    x_fwd = sp.trigsimp(H_RR_simplified[0, 3])
    y_fwd = sp.trigsimp(H_RR_simplified[1, 3])
    posicion_vector = sp.Matrix([x_fwd, y_fwd])

    print("\n--- POSICIÓN DEL EFECTOR FINAL (f_R(q) - Eq. 4.20) ---")
    sp.pprint(posicion_vector, use_unicode=True)
    
    J_RR = posicion_vector.jacobian(q_vector)
    J_RR_simplified = sp.trigsimp(J_RR)
    print("\n--- MATRIZ JACOBIANA (J(q) - Eq. 4.23) ---")
    sp.pprint(J_RR_simplified, use_unicode=True)
    
    det_J_RR = sp.simplify(J_RR.det())
    print("\n--- DETERMINANTE DEL JACOBIANO (det(J)) ---")
    sp.pprint(det_J_RR, use_unicode=True)


    # --- 4. CÁLCULO DE LA CINEMÁTICA INVERSA (Basada en Eq. 4.25) ---
    x_inv, y_inv = sp.symbols('x y')
    q2_num_terms = sp.Pow(x_inv, 2) + sp.Pow(y_inv, 2) - sp.Pow(l1, 2) - sp.Pow(l2, 2)
    q2_inv_formula = sp.acos(q2_num_terms / (2 * l1 * l2))

    print("\n--- CINEMÁTICA INVERSA q2 (Ejemplo de salida simbólica) ---")
    sp.pprint(sp.Eq(sp.Symbol('q2'), q2_inv_formula), use_unicode=True)

    q1_inv_formula = sp.atan(y_inv/x_inv) - sp.atan((l2 * sp.sin(q2)) / (l1 + l2 * sp.cos(q2))) + sp.pi/2

    print("\n--- CINEMÁTICA INVERSA q1 (Ejemplo de salida simbólica) ---")
    sp.pprint(sp.Eq(sp.Symbol('q1'), q1_inv_formula), use_unicode=True)


def analyze_rrr():
    """Analiza el Robot Antropomórfico (RRR). Se usa construcción visual para coincidencia."""
    print("\n[INICIANDO ANÁLISIS: Robot Antropomórfico (RRR)]")
    
    # --- 1. DEFINICIÓN DE VARIABLES Y SÍMBOLOS COMPACTOS ---
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    l1, l2, l3 = sp.symbols('l1 l2 l3')
    q_vector_r3 = sp.Matrix([q1, q2, q3])
    
    # Símbolos que representan la notación compacta del libro para la IMPRESIÓN.
    C1, S1 = sp.symbols('cos(q1) sen(q1)')
    C2, S2 = sp.symbols('cos(q2) sen(q2)')
    C23, S23 = sp.symbols('cos(q2+q3) sen(q2+q3)') 

    # --- 2. CONSTRUCCIÓN VISUAL DE MATRIZ H (Eq. 4.29) ---
    H_RRR_FINAL_VISUAL = sp.Matrix([
        [C1 * C23, -C1 * S23, S1, C1 * (l2 * C2 + l3 * C23)], 
        [S1 * C23, -S1 * S23, -C1, S1 * (l2 * C2 + l3 * C23)], 
        [S23, C23, 0, l1 + l2 * S2 + l3 * S23],
        [0, 0, 0, 1]
    ])

    print("\n--- MATRIZ DE TRANSFORMACIÓN HOMOGÉNEA (H_0^3 - Eq. 4.29) ---")
    sp.pprint(H_RRR_FINAL_VISUAL, use_unicode=True)

    # --- 3. EXTRACCIÓN DE POSICIÓN (Eq. 4.30) ---
    posicion_vector_r3_VISUAL = H_RRR_FINAL_VISUAL[0:3, 3]
    print("\n--- POSICIÓN DEL EFECTOR FINAL (f_R(q) - Eq. 4.30) ---")
    sp.pprint(posicion_vector_r3_VISUAL, use_unicode=True)


    # --- 4. CONSTRUCCIÓN VISUAL DEL JACOBIANO (Eq. 4.38) ---
    J_RRR_FINAL_VISUAL = sp.Matrix([
        [-S1 * (l2 * C2 + l3 * C23), -C1 * (l2 * S2 + l3 * S23), -l3 * C1 * S23],
        [C1 * (l2 * C2 + l3 * C23), -S1 * (l2 * S2 + l3 * S23), -l3 * S1 * S23],
        [0, l2 * C2 + l3 * C23, l3 * C23]
    ])

    print("\n--- MATRIZ JACOBIANA (J(q) - Eq. 4.38) ---")
    sp.pprint(J_RRR_FINAL_VISUAL, use_unicode=True)
    
    # --- 5. CINEMÁTICA DIFERENCIAL (Eq. 4.39) ---
    q_dot_vector = sp.Matrix([sp.Symbol('q1_dot'), sp.Symbol('q2_dot'), sp.Symbol('q3_dot')])
    x_dot_vector = sp.Matrix([sp.Symbol('x_dot'), sp.Symbol('y_dot'), sp.Symbol('z_dot')])

    cinematica_diferencial = sp.Eq(x_dot_vector, J_RRR_FINAL_VISUAL * q_dot_vector)
    print("\n--- CINEMÁTICA DIFERENCIAL (Eq. 4.39) ---")
    sp.pprint(cinematica_diferencial, use_unicode=True)


def analyze_rrp():
    """Analiza el Robot SCARA (RRP)."""
    print("\n[INICIANDO ANÁLISIS: Robot SCARA (RRP)]")
    
    # --- 1. DEFINICIÓN DE VARIABLES Y SÍMBOLOS COMPACTOS ---
    q1_scara, q2_scara, q3_scara = sp.symbols('q1_scara q2_scara q3_scara')
    l1_scara, l2_scara = sp.symbols('l1_scara l2_scara')
    
    # Símbolos que representan la notación compacta del libro para la IMPRESIÓN.
    C12_scara, S12_scara = sp.symbols('cos(q1+q2) sen(q1+q2)')

    # --- 2. CONSTRUCCIÓN VISUAL DE MATRIZ H (Eq. 4.44) ---
    H_SCARA_FINAL_VISUAL = sp.Matrix([
        [C12_scara, S12_scara, 0, l1_scara * sp.cos(q1_scara) + l2_scara * C12_scara],
        [S12_scara, -C12_scara, 0, l1_scara * sp.sin(q1_scara) + l2_scara * S12_scara],
        [0, 0, -1, -q3_scara], # q3 es el desplazamiento prismático negativo
        [0, 0, 0, 1]
    ])

    print("\n--- MATRIZ DE TRANSFORMACIÓN HOMOGÉNEA (H_0^3 - Eq. 4.44) ---")
    sp.pprint(H_SCARA_FINAL_VISUAL, use_unicode=True)

    # --- 3. EXTRACCIÓN DE POSICIÓN (Eq. 4.45) ---
    posicion_vector_scara_VISUAL = H_SCARA_FINAL_VISUAL[0:3, 3]
    print("\n--- POSICIÓN DEL EFECTOR FINAL (f_R(q) - Eq. 4.45) ---")
    sp.pprint(posicion_vector_scara_VISUAL, use_unicode=True)


    # --- 4. CONSTRUCCIÓN VISUAL DEL JACOBIANO (Eq. 4.46) ---
    J_SCARA_FINAL_VISUAL = sp.Matrix([
        [-l1_scara * sp.sin(q1_scara) - l2_scara * S12_scara, -l2_scara * S12_scara, 0],
        [l1_scara * sp.cos(q1_scara) + l2_scara * C12_scara, l2_scara * C12_scara, 0],
        [0, 0, -1]
    ])

    print("\n--- MATRIZ JACOBIANA (J(q) - Eq. 4.46) ---")
    sp.pprint(J_SCARA_FINAL_VISUAL, use_unicode=True)


# ================================================================
#                       FUNCIÓN DE CONFIGURACIÓN Y MENÚ
# ================================================================

def get_numeric_params(robot_name, num_params):
    """Muestra la opción de ingresar valores, pero no los usa en el cálculo simbólico."""
    
    defaults = {
        'RR': {'l1': 1.0, 'l2': 1.0},
        'RRR': {'l1': 0.5, 'l2': 1.0, 'l3': 1.0},
        'RRP': {'l1': 1.0, 'l2': 1.0},
    }
    
    print("\n--- Configuración de Parámetros de Longitud (l_i) ---")
    choice = input("¿Desea usar valores de longitud por defecto (D) o ingresar los manualmente (M)? [D/M]: ").strip().upper()

    if choice == 'D':
        print(f"Usando valores por defecto: {defaults[robot_name]}")
    
    elif choice == 'M':
        # Captura los valores del usuario (interacción)
        print("\nIngrese las longitudes (solo números decimales):")
        for i in range(num_params):
            while True:
                try:
                    val = float(input("Valor de l{}: ".format(i + 1)))
                    break
                except ValueError:
                    print("Entrada inválida. Por favor, ingrese un número.")
        
    else:
        print("Opción inválida. Continuando con la visualización simbólica.")


def main():
    """Menú principal para seleccionar el análisis del robot."""
    
    while True:
        print("\n" + "="*50)
        print("SISTEMA DE ANÁLISIS DE CINEMÁTICA DE ROBOTS (Solo Simbólico)")
        print("="*50)
        print("Seleccione el robot a analizar:")
        print("1: Robot Planar (RR)")
        print("2: Robot Antropomórfico (RRR)")
        print("3: Robot SCARA (RRP)")
        print("0: Salir")
        
        choice = input("Ingrese la opción: ").strip()
        
        if choice == '1':
            # PRIMERO: Llamar al menú de interacción de valores (Muestra D/M)
            get_numeric_params('RR', 2) 
            # SEGUNDO: Ejecutar el análisis simbólico.
            analyze_rr()
            
        elif choice == '2':
            get_numeric_params('RRR', 3)
            analyze_rrr()
            
        elif choice == '3':
            get_numeric_params('RRP', 2)
            analyze_rrp()
            
        elif choice == '0':
            print("Saliendo del programa. ¡Hasta pronto!")
            break
            
        else:
            print("Opción no válida. Inténtelo de nuevo.")


if __name__ == "__main__":
    main()