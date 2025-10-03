import sympy as sp
import numpy as np
import math
# ASUME: La clase ForwardKinematicsDH debe estar definida en 'forward_kinematics_dh_class.py'
from forward_kinematics_dh_class import ForwardKinematicsDH

# --- Configuración de parámetros por robot ---
PARAM_COUNTS = {
    'RR': {'links': 2, 'joints': 2},
    'RRR': {'links': 3, 'joints': 3},
    # RRP (SCARA) tiene 2 longitudes (l1, l2) y 3 juntas (q1, q2, q3)
    'RRP': {'links': 2, 'joints': 3} 
}

# ================================================================
#       FUNCIONES AUXILIARES DE CÁLCULO
# ================================================================

def _get_substitution_dictionaries(robot_name):
    """Genera los mapeos de la notación compacta (ej. C1) a la expresión real (cos(q1))."""
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    
    # 1. Sustituciones para RRR y RRP: Deshacer la notación compacta visual.
    sustituciones_base = {
        sp.Symbol('cos(q1)'): sp.cos(q1), sp.Symbol('sen(q1)'): sp.sin(q1),
        sp.Symbol('cos(q2)'): sp.cos(q2), sp.Symbol('sen(q2)'): sp.sin(q2),
    }

    if robot_name == 'RRR':
        # C23 = cos(q2+q3)
        sustituciones_base.update({
            sp.Symbol('cos(q2+q3)'): sp.cos(q2 + q3), 
            sp.Symbol('sen(q2+q3)'): sp.sin(q2 + q3)
        })
    elif robot_name == 'RRP':
        # C12 = cos(q1+q2)
        sustituciones_base.update({
            sp.Symbol('cos(q1+q2)'): sp.cos(q1 + q2), 
            sp.Symbol('sen(q1+q2)'): sp.sin(q1 + q2)
        })
        
    return sustituciones_base


def _perform_numerical_calc(robot_name, P_sym, J_sym, l_values, q_values):
    """
    Realiza la sustitución numérica en las matrices de Posición (P_sym) y Jacobiano (J_sym).
    """
    if not l_values or not q_values:
        return

    print("\n\n#################################################")
    print(f"--- CÁLCULO NUMÉRICO ({robot_name}) ---")
    print(f"L (Longitudes/Offsets): {l_values}")
    # Nota: Los valores de q se imprimen en radianes, pero se muestran los grados en la entrada.
    print(f"Q (Juntas en Radianes): {q_values}")
    print("#################################################")

    q_syms = [sp.Symbol(f'q{i+1}') for i in range(PARAM_COUNTS[robot_name]['joints'])]
    l_syms = [sp.Symbol(f'l{i+1}') for i in range(PARAM_COUNTS[robot_name]['links'])]
    
    # 1. Crear el diccionario de sustituciones numéricas (simbolo -> valor numerico)
    sustituciones_l = {l_syms[i]: l_values[f'l{i+1}'] for i in range(len(l_syms))}
    sustituciones_q = {q_syms[i]: q_values[f'q{i+1}'] for i in range(len(q_syms))}
    sustituciones_num = {**sustituciones_l, **sustituciones_q}

    # 2. Deshacer la notación compacta visual (si aplica)
    sustituciones_simbolicas = _get_substitution_dictionaries(robot_name)
    
    P_real_sym = P_sym.subs(sustituciones_simbolicas)
    J_real_sym = J_sym.subs(sustituciones_simbolicas)

    # 3. Aplicar las sustituciones numéricas y evaluar
    P_num = P_real_sym.subs(sustituciones_num).evalf()
    J_num = J_real_sym.subs(sustituciones_num).evalf()

    print("\nPOSICIÓN NUMÉRICA (P):")
    sp.pprint(P_num, use_unicode=True)

    print("\nJACOBIANO NUMÉRICO (J):")
    sp.pprint(J_num, use_unicode=True)


# ================================================================
#       FUNCIONES DE ANÁLISIS
# ================================================================

def analyze_rr(l_values=None, q_values=None, perform_numeric=False):
    """Analiza el Robot Planar (RR). Calcula H, Posición, Jacobiano y el valor NUMÉRICO."""
    print("\n[INICIANDO ANÁLISIS: Robot Planar (RR)]")

    q1, q2 = sp.symbols('q1 q2')
    l1, l2 = sp.symbols('l1 l2')
    
    dh_params_RR_sym = [
        [q1, 0, l1, 0],
        [q2, 0, l2, 0],
    ]

    # CÁLCULO SIMBÓLICO
    H_sym_RR = ForwardKinematicsDH.symbolic(dh_params_RR_sym)
    H_RR_simplified = sp.trigsimp(H_sym_RR)
    
    x_fwd = sp.trigsimp(H_RR_simplified[0, 3])
    y_fwd = sp.trigsimp(H_RR_simplified[1, 3])
    posicion_vector = sp.Matrix([x_fwd, y_fwd])
    q_vector = sp.Matrix([q1, q2])
    J_RR = posicion_vector.jacobian(q_vector)
    J_RR_simplified = sp.trigsimp(J_RR)
    
    print("\n--- POSICIÓN SIMBÓLICA (f_R(q)) ---")
    sp.pprint(posicion_vector, use_unicode=True)
    print("\n--- JACOBIANO SIMBÓLICO (J(q)) ---")
    sp.pprint(J_RR_simplified, use_unicode=True)

    # CÁLCULO NUMÉRICO CONDICIONAL
    if perform_numeric:
        _perform_numerical_calc('RR', posicion_vector, J_RR_simplified, l_values, q_values)
    
    # El resto del código simbólico de cinemática inversa se mantiene sin cambios
    # ...


def analyze_rrr(l_values=None, q_values=None, perform_numeric=False):
    """Analiza el Robot Antropomórfico (RRR) y calcula el valor NUMÉRICO."""
    print("\n[INICIANDO ANÁLISIS: Robot Antropomórfico (RRR)]")
    
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    l1, l2, l3 = sp.symbols('l1 l2 l3')
    
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
    print("\n--- MATRIZ DE TRANSFORMACIÓN HOMOGÉNEA VISUAL (H_0^3 - Eq. 4.29) ---")
    sp.pprint(H_RRR_FINAL_VISUAL, use_unicode=True)

    # --- 3. EXTRACCIÓN DE POSICIÓN (Eq. 4.30) ---
    posicion_vector_r3_VISUAL = H_RRR_FINAL_VISUAL[0:3, 3]
    print("\n--- POSICIÓN SIMBÓLICA VISUAL (f_R(q) - Eq. 4.30) ---")
    sp.pprint(posicion_vector_r3_VISUAL, use_unicode=True)

    # --- 4. CONSTRUCCIÓN VISUAL DEL JACOBIANO (Eq. 4.38) ---
    J_RRR_FINAL_VISUAL = sp.Matrix([
        [-S1 * (l2 * C2 + l3 * C23), -C1 * (l2 * S2 + l3 * S23), -l3 * C1 * S23],
        [C1 * (l2 * C2 + l3 * C23), -S1 * (l2 * S2 + l3 * S23), -l3 * S1 * S23],
        [0, l2 * C2 + l3 * C23, l3 * C23]
    ])
    print("\n--- MATRIZ JACOBIANA VISUAL (J(q) - Eq. 4.38) ---")
    sp.pprint(J_RRR_FINAL_VISUAL, use_unicode=True)
    
    # CÁLCULO NUMÉRICO CONDICIONAL
    if perform_numeric:
        _perform_numerical_calc('RRR', posicion_vector_r3_VISUAL, J_RRR_FINAL_VISUAL, l_values, q_values)


def analyze_rrp(l_values=None, q_values=None, perform_numeric=False):
    """Analiza el Robot SCARA (RRP) y calcula el valor NUMÉRICO."""
    print("\n[INICIANDO ANÁLISIS: Robot SCARA (RRP)]")
    
    # Se usan los nombres genéricos q1, l1 para compatibilidad con la función de cálculo numérico
    q1, q2, q3 = sp.symbols('q1 q2 q3')
    l1, l2 = sp.symbols('l1 l2')
    
    C12, S12 = sp.symbols('cos(q1+q2) sen(q1+q2)')

    # --- 2. CONSTRUCCIÓN VISUAL DE MATRIZ H (Eq. 4.44) ---
    H_SCARA_FINAL_VISUAL = sp.Matrix([
        [C12, S12, 0, l1 * sp.cos(q1) + l2 * C12],
        [S12, -C12, 0, l1 * sp.sin(q1) + l2 * S12],
        [0, 0, -1, -q3], # q3 es el desplazamiento prismático negativo
        [0, 0, 0, 1]
    ])
    print("\n--- MATRIZ DE TRANSFORMACIÓN HOMOGÉNEA VISUAL (H_0^3 - Eq. 4.44) ---")
    sp.pprint(H_SCARA_FINAL_VISUAL, use_unicode=True)

    # --- 3. EXTRACCIÓN DE POSICIÓN (Eq. 4.45) ---
    posicion_vector_scara_VISUAL = H_SCARA_FINAL_VISUAL[0:3, 3]
    print("\n--- POSICIÓN SIMBÓLICA VISUAL (f_R(q) - Eq. 4.45) ---")
    sp.pprint(posicion_vector_scara_VISUAL, use_unicode=True)

    # --- 4. CONSTRUCCIÓN VISUAL DEL JACOBIANO (Eq. 4.46) ---
    J_SCARA_FINAL_VISUAL = sp.Matrix([
        [-l1 * sp.sin(q1) - l2 * S12, -l2 * S12, 0],
        [l1 * sp.cos(q1) + l2 * C12, l2 * C12, 0],
        [0, 0, -1]
    ])
    print("\n--- MATRIZ JACOBIANA VISUAL (J(q) - Eq. 4.46) ---")
    sp.pprint(J_SCARA_FINAL_VISUAL, use_unicode=True)
    
    # CÁLCULO NUMÉRICO CONDICIONAL
    if perform_numeric:
        _perform_numerical_calc('RRP', posicion_vector_scara_VISUAL, J_SCARA_FINAL_VISUAL, l_values, q_values)


# ================================================================
#       FUNCIÓN DE CONFIGURACIÓN Y MENÚ
# ================================================================

def get_numeric_params(robot_name):
    """
    Captura los valores de longitud (l) y las juntas (q) y una bandera para el cálculo numérico.
    """
    counts = PARAM_COUNTS[robot_name]
    l_values = {}
    q_values = {}
    perform_numeric_calc = False # Bandera por defecto
    
    defaults = {
        'RR': {'l1': 1.0, 'l2': 1.0},
        'RRR': {'l1': 0.5, 'l2': 1.0, 'l3': 1.0},
        'RRP': {'l1': 1.0, 'l2': 1.0},
    }
    
    # --- 1. Preguntar si se desea el cálculo numérico ---
    print("\n--- Configuración ---")
    choice_calc = input("¿Desea realizar la evaluación numérica después del cálculo simbólico? [S/N]: ").strip().upper()
    if choice_calc == 'S':
        perform_numeric_calc = True
    else:
        print("Continuando solo con el cálculo simbólico...")
        # Retorna valores vacíos, el programa principal lo manejará
        return {}, {}, False 

    # --- 2. Obtener Longitudes (l_i) (Solo si se eligió cálculo numérico) ---
    print("\n--- Configuración de Parámetros de Longitud (l_i) y Offsets ---")
    choice_l = input("¿Valores por defecto (D) o manuales (M)? [D/M]: ").strip().upper()

    if choice_l == 'D':
        l_values = defaults.get(robot_name, {})
    else:
        print("\nIngrese las longitudes (solo números decimales):")
        for i in range(counts['links']):
            sym_name = f'l{i + 1}'
            while True:
                try:
                    val = float(input(f"Valor de {sym_name}: "))
                    l_values[sym_name] = val
                    break
                except ValueError:
                    print("Entrada inválida. Por favor, ingrese un número.")

    # --- 3. Obtener Valores de las Juntas (q_i) ---
    print(f"\n--- Configuración de Valores de Juntas (q_i) ({counts['joints']} juntas) ---")
    for i in range(counts['joints']):
        joint_name = f'q{i + 1}'
        while True:
            try:
                # Para la junta prismática (q3 en RRP), se pide el desplazamiento en unidades de longitud.
                if robot_name == 'RRP' and i == 2:
                    val = float(input(f"Valor de {joint_name} (desplazamiento lineal): "))
                    q_values[joint_name] = val
                    break
                
                # Para juntas de revolución, se pide en grados y se convierte a radianes.
                val_grados = float(input(f"Valor de {joint_name} (en grados): "))
                q_values[joint_name] = math.radians(val_grados)
                break
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")

    return l_values, q_values, perform_numeric_calc


def main():
    """Menú principal para seleccionar el análisis del robot."""
    
    while True:
        print("\n" + "="*70)
        print("SISTEMA DE ANÁLISIS DE CINEMÁTICA DE ROBOTS (Simbólico + Numérico)")
        print("="*70)
        print("Seleccione el robot a analizar:")
        print("1: Robot Planar (RR)")
        print("2: Robot Antropomórfico (RRR)")
        print("3: Robot SCARA (RRP)")
        print("0: Salir")
        
        choice = input("Ingrese la opción: ").strip()
        
        if choice == '1':
            # Se recibe la nueva bandera booleana
            l_num, q_num, do_calc = get_numeric_params('RR') 
            analyze_rr(l_values=l_num, q_values=q_num, perform_numeric=do_calc)
            
        elif choice == '2':
            # Se recibe la nueva bandera booleana
            l_num, q_num, do_calc = get_numeric_params('RRR')
            analyze_rrr(l_values=l_num, q_values=q_num, perform_numeric=do_calc)
            
        elif choice == '3':
            # Se recibe la nueva bandera booleana
            l_num, q_num, do_calc = get_numeric_params('RRP')
            analyze_rrp(l_values=l_num, q_values=q_num, perform_numeric=do_calc)
            
        elif choice == '0':
            print("Saliendo del programa. ¡Hasta pronto!")
            break
            
        else:
            print("Opción no válida. Inténtelo de nuevo.")


if __name__ == "__main__":
    main()