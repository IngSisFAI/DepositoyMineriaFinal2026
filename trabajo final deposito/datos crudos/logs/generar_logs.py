"""
Genera archivos de logs en formato tipo log2.txt, con atributos alineados a
generar_transacciones.py. Un tercio de los IDs empiezan con TX-2024-4...
Label is_fraud balanceado 50% true / 50% false (para luego reetiquetar con
aclaracionSupervisor.txt). Genera versión limpia y sucia con solo 2 tipos
de errores fáciles de limpiar.
"""

import random
from pathlib import Path
from typing import List, Tuple

# =========================
# Configuración
# =========================
NUM_LOGS = 3000
ARCHIVO_SALIDA = "logs_generados.log"

# Un tercio de los IDs serán TX-2024-4xxxxxxx (luego todos fraudes por aclaración)
FRACCION_TX_2024_4 = 1 / 3

# Label balanceado 30% fraude / 70% no fraude
FRAUD_RATE = 0.10

# Porcentaje de logs que tendrán errores (logs "sucios")
PORCENTAJE_SUCIOS = 0.50  # 50% limpios, 50% sucios

# En sucios: probabilidad de aplicar cada tipo de error (por línea)
# Así ambas clases de error aparecen en el archivo y son fáciles de limpiar
PROB_ERROR_COMMA_AMOUNT = 0.5   # Error 1: Amount con coma (1200,50)
PROB_ERROR_TYPO_STATUS = 0.5    # Error 2: "Statuss" en vez de "Status"

# Atributos alineados a generar_transacciones.py
MERCHANT_CATEGORIES = [
    "electronics", "food", "travel", "clothing", "pharmacy", "gas",
    "groceries", "entertainment",
]
METHODS = ["Card", "Transfer"]
STATUSES = ["APPROVED", "DECLINED", "FLAGGED"]


def generar_transaction_hour() -> int:
    """Hora de la transacción (0-23)"""
    return random.randint(0, 23)


def generar_id_tx_2024_4(contador: int) -> str:
    """ID en rango TX-2024-4xxxxxxx (8 dígitos, empieza con 4)."""
    # 40000001 .. 49999999
    n = random.randint(40000001, 49999999)
    return f"TX-2024-{n}"


def generar_id_otro(contador: int) -> str:
    """ID en rango TX-2024-00xxxxxx u otro (no 4...)."""
    n = random.randint(1, 39999999)
    return f"TX-2024-{str(n).zfill(8)}"


def generar_ip() -> str:
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def generar_linea(
    contador: int,
    es_tx_2024_4: bool,
    is_fraud: bool,
    aplicar_error_coma: bool,
    aplicar_error_typo: bool,
) -> str:
    transaction_hour = generar_transaction_hour()
    if es_tx_2024_4:
        tx_id = generar_id_tx_2024_4(contador)
    else:
        tx_id = generar_id_otro(contador)

    user = random.randint(100, 9999)
    amount = round(random.uniform(10.0, 150000.0), 2)

    # Formato Amount: limpio o con coma (error 1)
    if aplicar_error_coma:
        amount_str = f"{int(amount)},{int((amount % 1) * 100):02d}"
    else:
        amount_str = f"{amount:.2f}"

    merchant = random.choice(MERCHANT_CATEGORIES)
    method = random.choice(METHODS)
    status = random.choice(STATUSES)
    
    # Campos adicionales alineados con generar_transacciones.py
    foreign_transaction = random.choice([0, 1])
    location_mismatch = random.choice([0, 1])
    device_trust_score = random.randint(0, 100)
    velocity_last_24h = round(random.uniform(0.0, 10.0), 1)
    cardholder_age = random.randint(18, 80)

    # Key Status: "Status" o "Statuss" (error 2)
    status_key = "Statuss" if aplicar_error_typo else "Status"

    fraud_label = "true" if is_fraud else "false"

    partes = [
        f"transaction_hour={transaction_hour}",
        tx_id,
        f"User={user}",
        f"Amount={amount_str}",
        "ARS",
        f"MerchantCategory={merchant}",
        f"Method={method}",
        f"{status_key}={status}",
        f"ForeignTransaction={foreign_transaction}",
        f"LocationMismatch={location_mismatch}",
        f"DeviceTrustScore={device_trust_score}",
        f"VelocityLast24h={velocity_last_24h}",
        f"CardholderAge={cardholder_age}",
        f"IsFraud={fraud_label}",
    ]

    if random.random() < 0.7:
        partes.append(f"IP={generar_ip()}")

    return " ".join(partes)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    num_tx_4 = int(NUM_LOGS * FRACCION_TX_2024_4)
    num_otros = NUM_LOGS - num_tx_4

    # Índices que serán TX-2024-4... (un tercio)
    indices_tx_4 = set(random.sample(range(NUM_LOGS), num_tx_4))

    # Label balanceado según FRAUD_RATE (30% fraude, 70% no fraude)
    num_fraude = int(NUM_LOGS * FRAUD_RATE)
    num_no_fraude = NUM_LOGS - num_fraude
    fraud_flags: List[bool] = [False] * num_no_fraude + [True] * num_fraude
    random.shuffle(fraud_flags)

    # Determinar qué líneas serán "sucias" (con errores)
    num_sucias = int(NUM_LOGS * PORCENTAJE_SUCIOS)
    indices_sucias = set(random.sample(range(NUM_LOGS), num_sucias))

    lineas: List[str] = []

    for i in range(NUM_LOGS):
        es_tx_2024_4 = i in indices_tx_4
        is_fraud = fraud_flags[i]
        es_sucia = i in indices_sucias

        if es_sucia:
            # Aplicar errores a esta línea
            err_coma = random.random() < PROB_ERROR_COMMA_AMOUNT
            err_typo = random.random() < PROB_ERROR_TYPO_STATUS
            lineas.append(
                generar_linea(i, es_tx_2024_4, is_fraud, aplicar_error_coma=err_coma, aplicar_error_typo=err_typo)
            )
        else:
            # Línea limpia, sin errores
            lineas.append(
                generar_linea(i, es_tx_2024_4, is_fraud, aplicar_error_coma=False, aplicar_error_typo=False)
            )

    out_archivo = script_dir / ARCHIVO_SALIDA

    with out_archivo.open("w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
        f.write("\n")

    n_fraud = sum(1 for b in fraud_flags if b)
    print(f"Generados {NUM_LOGS} logs en un solo archivo.")
    print(f"  TX-2024-4...: {num_tx_4} ({100 * num_tx_4 / NUM_LOGS:.1f}%)")
    print(f"  IsFraud=true: {n_fraud}, IsFraud=false: {NUM_LOGS - n_fraud}")
    print(f"  Logs limpios: {NUM_LOGS - num_sucias} ({100 * (NUM_LOGS - num_sucias) / NUM_LOGS:.1f}%)")
    print(f"  Logs sucios: {num_sucias} ({100 * num_sucias / NUM_LOGS:.1f}%)")
    print(f"  Archivo: {out_archivo}")
    print("  Errores en logs sucios:")
    print("    (1) Amount con coma decimal (ej: 1200,50)")
    print("    (2) 'Statuss' en vez de 'Status'")


if __name__ == "__main__":
    random.seed(42)
    main()
