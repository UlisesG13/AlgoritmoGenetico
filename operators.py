"""
SAFE-CARGO | operators.py
Operadores geneticos: seleccion, cruce y mutacion.

Trabaja exclusivamente sobre objetos Cromosoma/Gen de chromosome.py.
Ninguna logica de aptitud vive aqui — eso es responsabilidad de fitness.py.

Operadores implementados
───────────────────────────────────────────────────────────────────────────
Seleccion:
    - seleccion_torneo     : torneo de k participantes (recomendado)
    - seleccion_ruleta     : ruleta proporcional a aptitud invertida

Cruce (retorna DOS hijos):
    - cruce_uniforme       : cada gen se hereda de padre1 o padre2 con prob 0.5
    - cruce_un_punto       : parte el cromosoma en un punto y combina segmentos

Mutacion (modifica in-place):
    - mutar_cromosoma      : aplica los 5 operadores de mutacion por gen con
                             sus probabilidades individuales
    Operadores atomicos (tambien exportados para pruebas):
    - mutar_zona           : reasigna zona aleatoriamente
    - mutar_posicion       : perturba x_ratio/y_ratio/z_ratio
    - mutar_orientacion    : gira theta_h o theta_v
    - mutar_amortiguador   : cambia tipo y/o caras activas
    - mutar_sujecion       : cambia modo de sujecion
───────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from copy import deepcopy

from data_model import (
    CaraItem, TipoAmortiguador, ModoSujecion, Orientacion,
)
from chromosome import Cromosoma, Gen


# ─────────────────────────────────────────────────────────────────────
# PARAMETROS DE MUTACION
# (probabilidades independientes para cada operador atomico)
# ─────────────────────────────────────────────────────────────────────

PROB_MUTAR_ZONA         = 0.08   # prob de reasignar zona a un gen
PROB_MUTAR_POSICION     = 0.15   # prob de perturbar posicion (x/y/z ratio)
PROB_MUTAR_ORIENTACION  = 0.10   # prob de cambiar theta_h o theta_v
PROB_MUTAR_AMORTIGUADOR = 0.08   # prob de cambiar tipo de buffer o caras
PROB_MUTAR_SUJECION     = 0.07   # prob de cambiar modo de sujecion

# Magnitud maxima de la perturbacion de posicion (en unidades de ratio)
DELTA_POSICION_MAX = 0.25


# ─────────────────────────────────────────────────────────────────────
# SELECCION
# ─────────────────────────────────────────────────────────────────────

def seleccion_torneo(
    poblacion : list[Cromosoma],
    k         : int = 3,
) -> Cromosoma:
    """
    Seleccion por torneo: elige k individuos al azar y retorna el mejor.

    Requiere que todos los cromosomas tengan aptitud calculada.
    'Mejor' significa aptitud MAS BAJA (el problema es de minimizacion).

    Args:
        poblacion : lista de cromosomas evaluados.
        k         : tamanio del torneo. Mayor k => mayor presion selectiva.

    Returns:
        Referencia al cromosoma ganador (no es copia).
    """
    if k > len(poblacion):
        k = len(poblacion)

    participantes = random.sample(poblacion, k)
    return min(participantes, key=lambda c: c.aptitud if c.aptitud is not None else float("inf"))


def seleccion_ruleta(
    poblacion : list[Cromosoma],
) -> Cromosoma:
    """
    Seleccion por ruleta proporcional a aptitud invertida.

    Transforma aptitudes de minimizacion en pesos de seleccion:
        peso_i = 1 / (aptitud_i + epsilon)
    de forma que los cromosomas con MENOR aptitud tienen MAYOR peso.

    Args:
        poblacion : lista de cromosomas evaluados.

    Returns:
        Referencia al cromosoma seleccionado.
    """
    epsilon = 1e-6
    pesos = [
        1.0 / ((c.aptitud if c.aptitud is not None else float("inf")) + epsilon)
        for c in poblacion
    ]
    total = sum(pesos)
    if total == 0:
        return random.choice(poblacion)

    r = random.uniform(0, total)
    acumulado = 0.0
    for cromosoma, peso in zip(poblacion, pesos):
        acumulado += peso
        if acumulado >= r:
            return cromosoma
    return poblacion[-1]


# ─────────────────────────────────────────────────────────────────────
# CRUCE
# ─────────────────────────────────────────────────────────────────────

def cruce_uniforme(
    padre1 : Cromosoma,
    padre2 : Cromosoma,
    prob_herencia: float = 0.5,
) -> tuple[Cromosoma, Cromosoma]:
    """
    Cruce uniforme a nivel de gen.

    Para cada posicion i, el hijo1 hereda el gen de padre1 con
    probabilidad `prob_herencia` y el de padre2 con 1-prob_herencia.
    El hijo2 recibe el complemento.

    Este operador mezcla informacion de ambos padres en toda la longitud
    del cromosoma, util cuando no hay bloques de genes relacionados.

    Args:
        padre1, padre2  : cromosomas con la misma lista de items.
        prob_herencia   : probabilidad de heredar de padre1 en cada posicion.

    Returns:
        (hijo1, hijo2) — dos nuevos cromosomas independientes.
    """
    assert len(padre1) == len(padre2), \
        "Los padres deben tener el mismo numero de genes."

    genes_h1: list[Gen] = []
    genes_h2: list[Gen] = []

    for g1, g2 in zip(padre1.genes, padre2.genes):
        if random.random() < prob_herencia:
            genes_h1.append(g1.clonar())
            genes_h2.append(g2.clonar())
        else:
            genes_h1.append(g2.clonar())
            genes_h2.append(g1.clonar())

    hijo1 = Cromosoma(
        genes    = genes_h1,
        items    = padre1.items,
        vehiculo = padre1.vehiculo,
        trayecto = padre1.trayecto,
        aptitud  = None,            # el hijo aun no ha sido evaluado
    )
    hijo2 = Cromosoma(
        genes    = genes_h2,
        items    = padre1.items,
        vehiculo = padre1.vehiculo,
        trayecto = padre1.trayecto,
        aptitud  = None,
    )
    return hijo1, hijo2


def cruce_un_punto(
    padre1 : Cromosoma,
    padre2 : Cromosoma,
) -> tuple[Cromosoma, Cromosoma]:
    """
    Cruce de un punto.

    Elige un punto de corte aleatorio en [1, n-1] y combina:
        hijo1 = [padre1[:punto] | padre2[punto:]]
        hijo2 = [padre2[:punto] | padre1[punto:]]

    Util para preservar bloques de genes consecutivos que ya esten
    bien ubicados (ej. todos los alimentos juntos en la misma zona).

    Args:
        padre1, padre2 : cromosomas con la misma lista de items.

    Returns:
        (hijo1, hijo2) — dos nuevos cromosomas independientes.
    """
    n = len(padre1)
    assert n == len(padre2), "Los padres deben tener el mismo numero de genes."

    punto = random.randint(1, n - 1)

    genes_h1 = [g.clonar() for g in padre1.genes[:punto]] \
             + [g.clonar() for g in padre2.genes[punto:]]
    genes_h2 = [g.clonar() for g in padre2.genes[:punto]] \
             + [g.clonar() for g in padre1.genes[punto:]]

    hijo1 = Cromosoma(
        genes    = genes_h1,
        items    = padre1.items,
        vehiculo = padre1.vehiculo,
        trayecto = padre1.trayecto,
        aptitud  = None,
    )
    hijo2 = Cromosoma(
        genes    = genes_h2,
        items    = padre1.items,
        vehiculo = padre1.vehiculo,
        trayecto = padre1.trayecto,
        aptitud  = None,
    )
    return hijo1, hijo2


# ─────────────────────────────────────────────────────────────────────
# OPERADORES ATOMICOS DE MUTACION
# ─────────────────────────────────────────────────────────────────────

def mutar_zona(gen: Gen, ids_zona: list[str]) -> None:
    """
    Reasigna la zona del gen a cualquiera de las zonas disponibles
    (puede resultar en la misma zona; es correcto para el AG).
    """
    gen.zona_id = random.choice(ids_zona)


def mutar_posicion(gen: Gen, zona: "Zona") -> None:
    """
    Perturba la posicion local (x, y, z) del gen en cm.

    El delta maximo es DELTA_POSICION_MAX * dimension_de_zona en cada eje,
    de forma que la perturbacion es proporcional al espacio disponible.
    El resultado siempre queda dentro de [zona.x_min, zona.x_max] etc.

    Args:
        gen  : gen a mutar (in-place).
        zona : objeto Zona de la zona actualmente asignada al gen.
    """
    ancho = zona.x_max - zona.x_min
    largo = zona.y_max - zona.y_min
    alto  = zona.z_max - zona.z_min

    gen.x = _clamp(
        gen.x + random.uniform(-DELTA_POSICION_MAX * ancho, DELTA_POSICION_MAX * ancho),
        zona.x_min, zona.x_max
    )
    gen.y = _clamp(
        gen.y + random.uniform(-DELTA_POSICION_MAX * largo, DELTA_POSICION_MAX * largo),
        zona.y_min, zona.y_max
    )
    gen.z = _clamp(
        gen.z + random.uniform(-DELTA_POSICION_MAX * alto, DELTA_POSICION_MAX * alto),
        zona.z_min, zona.z_max
    )


def mutar_orientacion(gen: Gen) -> None:
    """
    Cambia theta_h o theta_v del gen (50% de probabilidad cada uno).
    """
    if random.random() < 0.5:
        gen.orientacion = Orientacion(
            theta_h = random.choice([0, 90, 180, 270]),
            theta_v = gen.orientacion.theta_v,
        )
    else:
        gen.orientacion = Orientacion(
            theta_h = gen.orientacion.theta_h,
            theta_v = random.choice([0, 90]),
        )


def mutar_amortiguador(gen: Gen, fragilidad: float) -> None:
    """
    Cambia el tipo de amortiguador y/o las caras activas.

    La fragilidad del item sesga la probabilidad de asignar
    un buffer mas grueso: items fragiles raramente quedan sin buffer.

    Args:
        gen         : gen a mutar (in-place).
        fragilidad  : fragilidad del item asociado [0.0, 1.0].
    """
    opciones_tipo = [
        TipoAmortiguador.NINGUNO,
        TipoAmortiguador.CARTON,
        TipoAmortiguador.PLASTICO_BURBUJA,
        TipoAmortiguador.ALGODON,
    ]
    # Probabilidad de NINGUNO inversamente proporcional a fragilidad
    pesos = [
        max(0.05, 1.0 - fragilidad),   # NINGUNO
        0.35,                           # CARTON
        0.35,                           # PLASTICO_BURBUJA
        0.25,                           # ALGODON
    ]
    gen.tipo_amortiguador = random.choices(opciones_tipo, weights=pesos, k=1)[0]

    if gen.tipo_amortiguador == TipoAmortiguador.NINGUNO:
        gen.caras_activas = []
    else:
        # Mutar la cantidad y seleccion de caras activas
        n_caras = random.randint(1, 4)
        gen.caras_activas = random.sample(list(CaraItem), n_caras)


def mutar_sujecion(gen: Gen, trayecto_score: float) -> None:
    """
    Cambia el modo de sujecion del gen.

    `trayecto_score` es un valor [0,1] derivado de las inclinaciones
    del trayecto; sesga hacia sujeciones mas completas en terrenos
    mas exigentes.

    Args:
        gen             : gen a mutar (in-place).
        trayecto_score  : severidad del trayecto en [0, 1].
    """
    opciones = [
        ModoSujecion.NINGUNA,
        ModoSujecion.LONGITUDINAL,
        ModoSujecion.LATERAL,
        ModoSujecion.COMPLETA,
    ]
    # En terrenos severos, NINGUNA tiene menor peso
    peso_ninguna = max(0.05, 1.0 - trayecto_score)
    pesos = [peso_ninguna, 0.30, 0.30, trayecto_score * 0.5 + 0.05]
    gen.modo_sujecion = random.choices(opciones, weights=pesos, k=1)[0]


# ─────────────────────────────────────────────────────────────────────
# MUTACION COMPLETA DE UN CROMOSOMA
# ─────────────────────────────────────────────────────────────────────

def mutar_cromosoma(
    cromosoma       : Cromosoma,
    prob_zona       : float = PROB_MUTAR_ZONA,
    prob_posicion   : float = PROB_MUTAR_POSICION,
    prob_orientacion: float = PROB_MUTAR_ORIENTACION,
    prob_amort      : float = PROB_MUTAR_AMORTIGUADOR,
    prob_sujecion   : float = PROB_MUTAR_SUJECION,
) -> None:
    """
    Aplica los 5 operadores de mutacion atomicos a cada gen del cromosoma.
    Cada operador se dispara de forma independiente segun su probabilidad.

    Modifica el cromosoma IN-PLACE y resetea su aptitud a None
    para forzar re-evaluacion.

    Args:
        cromosoma       : individuo a mutar.
        prob_zona       : probabilidad de mutar la zona de cada gen.
        prob_posicion   : probabilidad de perturbar la posicion de cada gen.
        prob_orientacion: probabilidad de cambiar la orientacion de cada gen.
        prob_amort      : probabilidad de cambiar el amortiguador de cada gen.
        prob_sujecion   : probabilidad de cambiar el modo de sujecion de cada gen.
    """
    ids_zona = [z.id for z in cromosoma.vehiculo.zonas]

    # Severidad del trayecto normalizada en [0, 1] para sesgar sujecion y buffer
    t = cromosoma.trayecto
    trayecto_score = min(
        (t.inclinacion_long + t.inclinacion_lat) / 40.0, 1.0
    )

    item_map = {it.id: it for it in cromosoma.items}
    zona_map = {z.id: z for z in cromosoma.vehiculo.zonas}
    mutado   = False

    for gen in cromosoma.genes:
        item = item_map.get(gen.item_id)
        fragilidad = item.fragilidad if item else 0.0

        if random.random() < prob_zona:
            mutar_zona(gen, ids_zona)
            # Reubicar dentro de la nueva zona para evitar desborde inmediato
            nueva_zona = zona_map.get(gen.zona_id)
            if nueva_zona:
                import random as _r
                gen.x = _r.uniform(nueva_zona.x_min, nueva_zona.x_max)
                gen.y = _r.uniform(nueva_zona.y_min, nueva_zona.y_max)
                gen.z = _r.uniform(nueva_zona.z_min, nueva_zona.z_max)
            mutado = True

        if random.random() < prob_posicion:
            zona_actual = zona_map.get(gen.zona_id)
            if zona_actual:
                mutar_posicion(gen, zona_actual)
            mutado = True

        if random.random() < prob_orientacion:
            mutar_orientacion(gen)
            mutado = True

        if random.random() < prob_amort:
            mutar_amortiguador(gen, fragilidad)
            mutado = True

        if random.random() < prob_sujecion:
            mutar_sujecion(gen, trayecto_score)
            mutado = True

    if mutado:
        cromosoma.aptitud = None   # invalidar para re-evaluacion


# ─────────────────────────────────────────────────────────────────────
# ELITISMO
# ─────────────────────────────────────────────────────────────────────

def aplicar_elitismo(
    poblacion_anterior : list[Cromosoma],
    poblacion_nueva    : list[Cromosoma],
    n_elite            : int = 2,
) -> list[Cromosoma]:
    """
    Preserva los `n_elite` mejores individuos de la generacion anterior
    en la nueva poblacion, reemplazando a los peores de la nueva.

    Garantiza que la aptitud del mejor individuo nunca empeora
    entre generaciones.

    Args:
        poblacion_anterior : generacion actual (evaluada).
        poblacion_nueva    : hijos generados (pueden no estar evaluados).
        n_elite            : cuantos individuos elite se preservan.

    Returns:
        Nueva lista con los elite insertados.
    """
    if n_elite <= 0:
        return poblacion_nueva

    # Ordenar anterior por aptitud (menor = mejor)
    elite = sorted(
        [c for c in poblacion_anterior if c.aptitud is not None],
        key=lambda c: c.aptitud
    )[:n_elite]

    # Ordenar nueva por aptitud descendente (los peores primero para reemplazar)
    nueva_ordenada = sorted(
        poblacion_nueva,
        key=lambda c: (c.aptitud if c.aptitud is not None else float("inf")),
        reverse=True
    )

    # Reemplazar los peores de la nueva con los elite
    for i, individuo_elite in enumerate(elite):
        nueva_ordenada[i] = individuo_elite.clonar()

    return nueva_ordenada


# ─────────────────────────────────────────────────────────────────────
# HELPER INTERNO
# ─────────────────────────────────────────────────────────────────────

def _clamp(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    """Acondiciona un valor al rango [minimo, maximo]."""
    return max(minimo, min(maximo, valor))


# ─────────────────────────────────────────────────────────────────────
# BLOQUE DE PRUEBA RAPIDA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from knowledge_base import CATALOGO_ITEMS, TRAYECTOS, crear_vehiculo_prueba
    from chromosome import crear_poblacion

    vehiculo     = crear_vehiculo_prueba()
    trayecto     = TRAYECTOS["severo"]
    items_prueba = CATALOGO_ITEMS[:6]

    print("=== SAFE-CARGO | operators.py — prueba ===\n")

    # Crear poblacion base (sin evaluar — aptitud = None)
    poblacion = crear_poblacion(20, items_prueba, vehiculo, trayecto)
    # Simular aptitudes aleatorias para poder probar seleccion
    for c in poblacion:
        c.aptitud = random.uniform(0.1, 5.0)

    # ── Seleccion ──
    ganador_torneo = seleccion_torneo(poblacion, k=3)
    ganador_ruleta = seleccion_ruleta(poblacion)
    print(f"Torneo (k=3): aptitud ganador = {ganador_torneo.aptitud:.4f}")
    print(f"Ruleta:       aptitud ganador = {ganador_ruleta.aptitud:.4f}")
    mejor_real = min(poblacion, key=lambda c: c.aptitud)
    print(f"Mejor real:   aptitud         = {mejor_real.aptitud:.4f}\n")

    # ── Cruce uniforme ──
    p1, p2 = poblacion[0], poblacion[1]
    h1, h2 = cruce_uniforme(p1, p2)
    print(f"Cruce uniforme:")
    print(f"  Padre1 zona[0]={p1.genes[0].zona_id}, Padre2 zona[0]={p2.genes[0].zona_id}")
    print(f"  Hijo1  zona[0]={h1.genes[0].zona_id}, Hijo2  zona[0]={h2.genes[0].zona_id}")
    print(f"  Hijo1 aptitud={h1.aptitud} (debe ser None)\n")

    # ── Cruce un punto ──
    h3, h4 = cruce_un_punto(p1, p2)
    print(f"Cruce un punto:")
    print(f"  Hijo3 genes[:3] zonas = {[g.zona_id for g in h3.genes[:3]]}")
    print(f"  Hijo4 genes[:3] zonas = {[g.zona_id for g in h4.genes[:3]]}\n")

    # ── Mutacion ──
    candidato = poblacion[0].clonar()
    zona_map  = {z.id: z for z in vehiculo.zonas}

    mutar_cromosoma(candidato,
                    prob_zona=1.0,
                    prob_posicion=1.0,
                    prob_orientacion=0.0,
                    prob_amort=0.0,
                    prob_sujecion=0.0)

    zona_despues = [g.zona_id for g in candidato.genes]
    print(f"Mutacion (zona + posicion forzadas):")
    print(f"  Zonas despues: {zona_despues}")
    for gen in candidato.genes:
        z = zona_map[gen.zona_id]
        assert z.x_min <= gen.x <= z.x_max, f"X fuera de rango en {gen.item_id}"
        assert z.y_min <= gen.y <= z.y_max, f"Y fuera de rango en {gen.item_id}"
        assert z.z_min <= gen.z <= z.z_max, f"Z fuera de rango en {gen.item_id}"
    print(f"  ✓ Todas las coords en limites de zona tras mutacion\n")

    # ── Elitismo ──
    nueva_gen = crear_poblacion(20, items_prueba, vehiculo, trayecto)
    for c in nueva_gen:
        c.aptitud = random.uniform(1.0, 10.0)

    mejor_antes = min(poblacion, key=lambda c: c.aptitud).aptitud
    con_elite   = aplicar_elitismo(poblacion, nueva_gen, n_elite=2)
    mejor_despues = min(c.aptitud for c in con_elite if c.aptitud is not None)
    print(f"Elitismo (n=2):")
    print(f"  Mejor aptitud generacion anterior : {mejor_antes:.4f}")
    print(f"  Mejor aptitud nueva generacion    : {mejor_despues:.4f}")
    assert mejor_despues <= mejor_antes + 1e-9, \
        "ERROR: el elitismo no preservo al mejor individuo"
    print(f"  ✓ Elitismo preserva al mejor correctamente\n")

    print("=== Prueba completada ===")
