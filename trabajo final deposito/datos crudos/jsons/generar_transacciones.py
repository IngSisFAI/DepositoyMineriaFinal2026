import copy
import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Configuración (editable)
# =========================
NUM_TRANSACCIONES = 5000
ARCHIVO_SALIDA = "transacciones_generadas.json"

# Generar "sucias" (valores faltantes / tipos raros / outliers) usando como base el CSV sucio
PORCENTAJE_SUCIAS = 0.10

# Permitir duplicados exactos (misma transacción repetida)
PORCENTAJE_DUPLICADAS = 0.03

# Balance de fraude. Si es None, usa el ratio real del CSV (~2.6%).
FRAUD_RATE_OBJETIVO: Optional[float] = 0.15

# Campo "casi nulo": presente solo en pocas transacciones y NO afecta el fraude
CAMPO_CASI_NULO = "schema_version"
PROB_CAMPO_CASI_NULO = 0.02

# Fuente para basar la relación fraude/no-fraude
CSV_BASE = Path(__file__).resolve().parents[1] / "csv_sucio" / "credit_card_fraud_10k_dirty.csv"

# Datos para completar campos que no están en el CSV
CIUDADES_ARGENTINA = [
    "Buenos Aires",
    "Córdoba",
    "Rosario",
    "Mendoza",
    "Tucumán",
    "La Plata",
    "Mar del Plata",
    "Salta",
    "Santa Fe",
    "San Juan",
    "Neuquén",
    "Bahía Blanca",
]
CIUDADES_EXTRANJERAS = ["Miami", "New York", "Madrid", "Barcelona", "São Paulo", "Lima", "Santiago"]
DISPOSITIVOS = ["mobile", "desktop", "tablet"]
SISTEMAS_OPERATIVOS = ["android", "ios", "windows", "linux", "macos"]
METODOS_AUTENTICACION = ["biometric", "pin", "password", "2fa"]
PAISES = ["AR", "US", "ES", "BR", "CL", "PE"]

_CATEGORIA_MAP = {
    "electronics": "electronics",
    "food": "food",
    "travel": "travel",
    "clothing": "clothing",
    "pharmacy": "pharmacy",
    "gas": "gas",
    "grocery": "groceries",
    "groceries": "groceries",
    "entertainment": "entertainment",
}


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" else None


def _to_float(x: Any) -> Optional[float]:
    s = _as_str(x)
    if s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def _to_int(x: Any) -> Optional[int]:
    v = _to_float(x)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_bool01(x: Any) -> Optional[bool]:
    v = _to_float(x)
    if v is None:
        return None
    return bool(v >= 0.5)


def _normalizar_categoria(x: Any) -> Optional[str]:
    s = _as_str(x)
    if s is None:
        return None
    key = s.strip().lower()
    return _CATEGORIA_MAP.get(key, key)


def generar_timestamp_aleatorio() -> str:
    inicio = datetime(2024, 1, 1)
    fin = datetime(2024, 12, 31, 23, 59, 59)
    delta = fin - inicio
    segundos_aleatorios = random.randint(0, int(delta.total_seconds()))
    fecha = inicio + timedelta(seconds=segundos_aleatorios)
    return fecha.strftime("%Y-%m-%dT%H:%M:%SZ")


def generar_id_transaccion(contador: int) -> str:
    return f"TX-2024-{str(contador).zfill(8)}"


def _row_es_cleanish(row: Dict[str, str]) -> bool:
    amount = _to_float(row.get("amount"))
    hour = _to_int(row.get("transaction_hour"))
    cat = _normalizar_categoria(row.get("merchant_category"))
    foreign = _to_float(row.get("foreign_transaction"))
    mismatch = _to_float(row.get("location_mismatch"))
    trust = _to_float(row.get("device_trust_score"))
    vel = _to_float(row.get("velocity_last_24h"))
    age = _to_float(row.get("cardholder_age"))

    if amount is None:
        return False
    if hour is None or not (0 <= hour <= 23):
        return False
    if cat is None:
        return False
    if foreign is None or foreign not in (0.0, 1.0):
        return False
    if mismatch is None or mismatch not in (0.0, 1.0):
        return False
    if trust is None or not (0.0 <= trust <= 100.0):
        return False
    if vel is None or vel < 0:
        return False
    if age is None or not (0 <= age <= 120):
        return False
    return True


def _parse_label(row: Dict[str, str]) -> Optional[int]:
    v = _to_float(row.get("is_fraud"))
    if v is None:
        return None
    return 1 if v >= 0.5 else 0


def cargar_csv_base(path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Devuelve (rows_legit, rows_fraud) SOLO con label válido (0/1)."""
    rows_legit: List[Dict[str, str]] = []
    rows_fraud: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _parse_label(row)
            if label is None:
                continue
            if label == 1:
                rows_fraud.append(row)
            else:
                rows_legit.append(row)
    return rows_legit, rows_fraud


def _calc_risk_score(
    amount: Optional[float],
    foreign: Optional[bool],
    mismatch: Optional[bool],
    trust_score_0_1: Optional[float],
    velocity: Optional[float],
) -> float:
    """
    Score (0..1) calculado SOLO desde features (no usa label).
    Sirve para que haya relación entre fraude/no-fraude para ML.
    """
    risk = 0.05
    if foreign is True:
        risk += 0.25
    if mismatch is True:
        risk += 0.25
    if amount is not None and amount >= 500:
        risk += 0.15
    if velocity is not None and velocity >= 10:
        risk += 0.10
    if trust_score_0_1 is not None and trust_score_0_1 <= 0.40:
        risk += 0.10
    risk += random.uniform(-0.05, 0.05)
    return max(0.0, min(1.0, risk))


def _make_auth(risk: float) -> Dict[str, Any]:
    if risk >= 0.65:
        method = random.choice(["password", "pin", "2fa"])
        failed_attempts = random.randint(1, 3)
    elif risk >= 0.35:
        method = random.choice(["pin", "2fa", "biometric"])
        failed_attempts = random.randint(0, 2)
    else:
        method = random.choice(["biometric", "pin"])
        failed_attempts = random.randint(0, 1)
    return {"method": method, "failed_attempts": failed_attempts}


def _maybe_dirty_value(want_dirty: bool, clean_value: Any, raw_value: Any) -> Any:
    """
    Si want_dirty=True, a veces devuelve el valor crudo del CSV (puede ser string raro).
    Si want_dirty=False, devuelve el valor limpio.
    """
    if not want_dirty:
        return clean_value
    # 70% se mantiene limpio, 30% se ensucia para que sea "dirty" pero no imposible de usar
    return clean_value if random.random() < 0.70 else (raw_value if raw_value is not None else None)


def row_a_transaccion_tipo1(row: Dict[str, str], contador: int, want_dirty: bool) -> Dict[str, Any]:
    # Parseo base desde CSV
    raw_amount = row.get("amount")
    raw_hour = row.get("transaction_hour")
    raw_cat = row.get("merchant_category")
    raw_foreign = row.get("foreign_transaction")
    raw_mismatch = row.get("location_mismatch")
    raw_trust = row.get("device_trust_score")
    raw_vel = row.get("velocity_last_24h")
    raw_age = row.get("cardholder_age")

    amount = _to_float(raw_amount)
    hour = _to_int(raw_hour)
    cat = _normalizar_categoria(raw_cat)
    foreign = _to_bool01(raw_foreign)
    mismatch = _to_bool01(raw_mismatch)
    trust_0_100 = _to_float(raw_trust)
    if trust_0_100 is not None:
        trust_0_100 = max(0.0, min(100.0, trust_0_100))
    trust_0_1 = None if trust_0_100 is None else trust_0_100 / 100.0
    velocity = _to_float(raw_vel)
    age = _to_int(raw_age)

    # Completar si faltan (para que no se rompa todo) — pero si want_dirty, se permite que queden feos
    if amount is None and not want_dirty:
        amount = round(random.uniform(10.0, 5000.0), 2)
    if hour is None:
        hour = random.randint(0, 23)
    if cat is None and not want_dirty:
        cat = random.choice(list(_CATEGORIA_MAP.values()))
    if foreign is None:
        foreign = random.random() < 0.10
    if mismatch is None:
        mismatch = random.random() < 0.20  # 20% de probabilidad (más realista)
    if trust_0_100 is None:
        trust_0_100 = float(random.randint(10, 100))
        trust_0_1 = trust_0_100 / 100.0
    elif trust_0_1 is None:
        trust_0_1 = 0.5
    if velocity is None:
        velocity = float(random.randint(0, 30))
    if age is None:
        age = random.randint(18, 80)

    # Score/riesgo (derivado de features, NO del label)
    risk = _calc_risk_score(amount, foreign, mismatch, trust_0_1, velocity)

    # Campos que no están en el CSV
    if foreign is True:
        city = random.choice(CIUDADES_EXTRANJERAS)
        country_loc = "United States" if city in {"Miami", "New York"} else "Argentina"
        cardholder_country = random.choice(["AR", "US", "ES", "BR", "CL", "PE"])
    else:
        city = random.choice(CIUDADES_ARGENTINA)
        country_loc = "Argentina"
        cardholder_country = random.choice(["AR", "AR", "AR", "US", "BR", "CL", "PE", "ES"])

    auth = _make_auth(risk)

    # ip_risk_score a partir de features (para ML)
    ip_risk_score = round(risk, 2)

    # Label desde CSV (para mantener relación real); si falta, caemos a un default razonable.
    label = _parse_label(row)
    is_fraud = bool(label == 1) if label is not None else False

    # Campo casi nulo (independiente del fraude)
    rare_value = None
    if random.random() < PROB_CAMPO_CASI_NULO:
        rare_value = 1.0

    # Campos que no deben ser nulos: usar valor sucio o fallback al limpio
    out_hour = _maybe_dirty_value(want_dirty, hour, raw_hour)
    out_age = _maybe_dirty_value(want_dirty, age, raw_age)
    out_trust = _maybe_dirty_value(want_dirty, trust_0_100, raw_trust)
    out_velocity = _maybe_dirty_value(want_dirty, velocity, raw_vel)
    out_foreign = _maybe_dirty_value(want_dirty, foreign, raw_foreign)
    out_mismatch = _maybe_dirty_value(want_dirty, mismatch, raw_mismatch)

    return {
        # Formato tipo transaccion1.json (un solo tipo)
        "transaction_id": generar_id_transaccion(contador),
        "timestamp": generar_timestamp_aleatorio(),
        "transaction_hour": out_hour if out_hour is not None else hour,
        "amount": _maybe_dirty_value(want_dirty, amount, raw_amount),
        "merchant_category": _normalizar_categoria(_maybe_dirty_value(want_dirty, cat, raw_cat)) or cat or "electronics",
        "cardholder": {
            "age": out_age if out_age is not None else age,
            "country": cardholder_country,
        },
        "device": {
            "device_type": random.choice(DISPOSITIVOS),
            "operating_system": random.choice(SISTEMAS_OPERATIVOS),
            "device_trust_score": out_trust if out_trust is not None else trust_0_100,
        },
        "network_features": {
            "velocity_last_24h": out_velocity if out_velocity is not None else velocity,
            "ip_risk_score": ip_risk_score,
        },
        "location": {
            "city": city,
            "country": country_loc,
            "is_foreign_transaction": out_foreign if out_foreign is not None else foreign,
            "location_mismatch": out_mismatch if out_mismatch is not None else mismatch,
        },
        "authentication": auth,
        "labels": {"is_fraud": is_fraud},
        CAMPO_CASI_NULO: rare_value,
    }


def main() -> None:
    print(f"Generando {NUM_TRANSACCIONES} transacciones (solo tipo transaccion1.json)...")
    if not CSV_BASE.exists():
        raise FileNotFoundError(f"No existe el CSV base: {CSV_BASE}")

    rows_legit, rows_fraud = cargar_csv_base(CSV_BASE)
    if not rows_legit or not rows_fraud:
        raise RuntimeError("CSV base no tiene suficientes filas con label 0/1 para muestrear.")

    # Separar pools "clean-ish" vs "dirty" para controlar PORCENTAJE_SUCIAS
    legit_clean = [r for r in rows_legit if _row_es_cleanish(r)]
    legit_dirty = [r for r in rows_legit if not _row_es_cleanish(r)]
    fraud_clean = [r for r in rows_fraud if _row_es_cleanish(r)]
    fraud_dirty = [r for r in rows_fraud if not _row_es_cleanish(r)]

    csv_fraud_rate = len(rows_fraud) / (len(rows_fraud) + len(rows_legit))
    target_fraud_rate = csv_fraud_rate if FRAUD_RATE_OBJETIVO is None else FRAUD_RATE_OBJETIVO

    print(f"- CSV base: legit={len(rows_legit)}, fraude={len(rows_fraud)}, rate={csv_fraud_rate:.4f}")
    print(f"- Objetivo fraude: {target_fraud_rate:.4f}")
    print(f"- Objetivo sucias: {PORCENTAJE_SUCIAS:.2%}")
    print(f"- Objetivo duplicadas: {PORCENTAJE_DUPLICADAS:.2%}")

    transacciones: List[Dict[str, Any]] = []

    for i in range(1, NUM_TRANSACCIONES + 1):
        # Duplicado exacto
        if transacciones and random.random() < PORCENTAJE_DUPLICADAS:
            transacciones.append(copy.deepcopy(random.choice(transacciones)))
            continue

        want_dirty = random.random() < PORCENTAJE_SUCIAS
        want_fraud = random.random() < target_fraud_rate

        if want_fraud:
            pool = fraud_dirty if want_dirty and fraud_dirty else fraud_clean if fraud_clean else rows_fraud
        else:
            pool = legit_dirty if want_dirty and legit_dirty else legit_clean if legit_clean else rows_legit

        row = random.choice(pool)
        transacciones.append(row_a_transaccion_tipo1(row, i, want_dirty=want_dirty))

        if i % 200 == 0:
            print(f"Generadas {i}/{NUM_TRANSACCIONES} transacciones...")

    out_path = Path(__file__).resolve().parent / ARCHIVO_SALIDA
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(transacciones, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nOK: Archivo generado exitosamente: {out_path}")
    print(f"  Total de transacciones: {len(transacciones)}")
    print("  Formato: solo tipo transaccion1.json")


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()

