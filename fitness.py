"""
SAFE-CARGO | fitness.py
Funcion de aptitud del AG.
Evalua cada cromosoma calculando las 5 metricas de optimizacion
y penalizaciones duras por violaciones estructurales.

f = w1*Dc + w2*Vr + w3*Pr + w4*Dv + w5*Pf  +  penalizaciones_duras
"""

import math
from dataclasses import dataclass
from data_model import (
    Item, Zona, Vehiculo, Trayecto,
    TipoAmortiguador, Categoria, ModoSujecion,
    AMORTIGUADORES
)
from chromosome import Cromosoma, Gen
from knowledge_base import son_compatibles


# ─────────────────────────────────────────────
# PESOS DE LA FUNCION DE APTITUD
# ─────────────────────────────────────────────

@dataclass
class PesosAptitud:
    w_dc : float = 0.35   # desplazamiento centro de masa
    w_vr : float = 0.25   # varianza distribucion de peso
    w_pr : float = 0.15   # penalizacion vecindad incompatible
    w_dv : float = 0.15   # desperdicio volumetrico
    w_pf : float = 0.10   # presion sobre mercancia fragil


PESOS_DEFAULT = PesosAptitud()

# Penalizaciones duras — valores altos para que el AG las evite siempre
PENALIZACION_COLISION          = 500.0
PENALIZACION_DESBORDE          = 300.0
PENALIZACION_ZONA_PROHIBIDA    = 400.0
PENALIZACION_PESO_ZONA         = 200.0
PENALIZACION_PESO_TOTAL        = 500.0

# Radio de proximidad para considerar dos items como "vecinos" (cm)
RADIO_VECINDAD = 5.0


# ─────────────────────────────────────────────
# HELPERS — GEOMETRIA
# ─────────────────────────────────────────────

def dimensiones_con_amortiguador(
    gen: Gen,
    item: Item
) -> tuple[float, float, float]:
    """
    Retorna las dimensiones efectivas del item ya rotado
    mas el grosor del amortiguador en las caras activas.
    """
    l_rot, a_rot, h_rot = gen.orientacion.aplicar(
        item.largo, item.ancho, item.alto
    )
    if gen.tipo_amortiguador == TipoAmortiguador.NINGUNO:
        return (l_rot, a_rot, h_rot)

    amor = AMORTIGUADORES[gen.tipo_amortiguador]
    from data_model import CaraItem
    dx = amor.grosor * (CaraItem.LEFT  in gen.caras_activas) \
       + amor.grosor * (CaraItem.RIGHT in gen.caras_activas)
    dy = amor.grosor * (CaraItem.FRONT in gen.caras_activas) \
       + amor.grosor * (CaraItem.BACK  in gen.caras_activas)
    dz = amor.grosor * (CaraItem.TOP    in gen.caras_activas) \
       + amor.grosor * (CaraItem.BOTTOM in gen.caras_activas)

    return (l_rot + dy, a_rot + dx, h_rot + dz)


def bbox_absoluto(
    gen: Gen,
    item: Item,
    zona: Zona
) -> tuple[float, float, float, float, float, float]:
    """
    Retorna el bounding box absoluto del item con amortiguador:
    (x_min, x_max, y_min, y_max, z_min, z_max)
    en coordenadas globales del vehiculo.
    """
    l_ef, a_ef, h_ef = dimensiones_con_amortiguador(gen, item)
    x_abs, y_abs, z_abs = zona.a_absoluto(gen.x, gen.y, gen.z)

    return (
        x_abs,         x_abs + a_ef,
        y_abs,         y_abs + l_ef,
        z_abs,         z_abs + h_ef,
    )


def hay_colision(
    bb1: tuple[float, float, float, float, float, float],
    bb2: tuple[float, float, float, float, float, float]
) -> bool:
    """
    Detecta solapamiento entre dos bounding boxes 3D.
    bb = (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    return (bb1[0] < bb2[1] and bb1[1] > bb2[0] and
            bb1[2] < bb2[3] and bb1[3] > bb2[2] and
            bb1[4] < bb2[5] and bb1[5] > bb2[4])


def son_vecinos(
    bb1: tuple[float, float, float, float, float, float],
    bb2: tuple[float, float, float, float, float, float],
    radio: float = RADIO_VECINDAD
) -> bool:
    """
    Dos items son vecinos si sus bounding boxes estan
    a menos de `radio` cm en cualquier eje.
    """
    gap_x = max(0.0, max(bb1[0], bb2[0]) - min(bb1[1], bb2[1]))
    gap_y = max(0.0, max(bb1[2], bb2[2]) - min(bb1[3], bb2[3]))
    gap_z = max(0.0, max(bb1[4], bb2[4]) - min(bb1[5], bb2[5]))
    return (gap_x <= radio and gap_y <= radio and gap_z <= radio)


# ─────────────────────────────────────────────
# METRICAS INDIVIDUALES
# ─────────────────────────────────────────────

def calcular_Dc(
    genes: list[Gen],
    items_dict: dict[str, Item],
    zonas_dict: dict[str, Zona],
    vehiculo: Vehiculo
) -> float:
    """
    Dc — Desplazamiento del centro de masa.
    Distancia euclidiana entre el CM real de la carga
    y el centro geometrico estable del vehiculo (baricentro XY de la batea).
    Normalizado por la diagonal del vehiculo para que quede en [0, 1].
    """
    total_peso = 0.0
    cm_x = cm_y = cm_z = 0.0

    for gen in genes:
        item = items_dict[gen.item_id]
        zona = zonas_dict[gen.zona_id]
        l_ef, a_ef, h_ef = dimensiones_con_amortiguador(gen, item)
        x_abs, y_abs, z_abs = zona.a_absoluto(gen.x, gen.y, gen.z)

        # Centro del item en coordenadas absolutas
        cx = x_abs + a_ef / 2
        cy = y_abs + l_ef / 2
        cz = z_abs + h_ef / 2

        cm_x += item.peso * cx
        cm_y += item.peso * cy
        cm_z += item.peso * cz
        total_peso += item.peso

    if total_peso == 0:
        return 0.0

    cm_x /= total_peso
    cm_y /= total_peso
    cm_z /= total_peso

    # Punto optimo: centro geometrico XY del vehiculo, z minimo (carga baja = estable)
    # Calculado como centroide de todos los volumenes de zonas ponderados
    vol_total = vehiculo.volumen_total()
    opt_x = opt_y = 0.0
    for zona in vehiculo.zonas:
        peso_zona = zona.volumen() / vol_total
        opt_x += peso_zona * (zona.x_offset + (zona.x_max - zona.x_min) / 2)
        opt_y += peso_zona * (zona.y_offset + (zona.y_max - zona.y_min) / 2)
    opt_z = 0.0  # queremos la carga lo mas baja posible

    distancia = math.sqrt(
        (cm_x - opt_x) ** 2 +
        (cm_y - opt_y) ** 2 +
        (cm_z - opt_z) ** 2
    )

    # Normalizar por la diagonal maxima del vehiculo
    todos_x = [z.x_offset + z.x_max for z in vehiculo.zonas]
    todos_y = [z.y_offset + z.y_max for z in vehiculo.zonas]
    todos_z = [z.z_offset + z.z_max for z in vehiculo.zonas]
    diagonal = math.sqrt(max(todos_x)**2 + max(todos_y)**2 + max(todos_z)**2)

    return distancia / diagonal if diagonal > 0 else 0.0


def calcular_Vr(
    genes: list[Gen],
    items_dict: dict[str, Item],
    vehiculo: Vehiculo
) -> float:
    """
    Vr — Varianza de distribucion de peso entre zonas.
    Normalizada por el cuadrado del peso promedio.
    """
    peso_por_zona = {z.id: 0.0 for z in vehiculo.zonas}

    for gen in genes:
        if gen.zona_id in peso_por_zona:
            peso_por_zona[gen.zona_id] += items_dict[gen.item_id].peso

    pesos = list(peso_por_zona.values())
    promedio = sum(pesos) / len(pesos) if pesos else 0.0

    if promedio == 0:
        return 0.0

    varianza = sum((p - promedio) ** 2 for p in pesos) / len(pesos)
    return varianza / (promedio ** 2)   # normalizada (coeficiente de variacion^2)


def calcular_Pr(
    genes: list[Gen],
    items_dict: dict[str, Item],
    zonas_dict: dict[str, Zona],
    bboxes: dict[str, tuple]
) -> float:
    """
    Pr — Penalizacion por vecindad incompatible.
    Por cada par de items vecinos con categorias incompatibles,
    se suma 1.0 al score. Normalizado por el total de pares posibles.
    """
    n = len(genes)
    if n < 2:
        return 0.0

    total_pares = n * (n - 1) / 2
    pares_invalidos = 0

    for i in range(n):
        for j in range(i + 1, n):
            g_i, g_j = genes[i], genes[j]
            # Solo penalizar si estan en la misma zona o son vecinos proximos
            if g_i.zona_id != g_j.zona_id:
                continue
            bb_i = bboxes[g_i.item_id]
            bb_j = bboxes[g_j.item_id]
            if son_vecinos(bb_i, bb_j):
                cat_i = items_dict[g_i.item_id].categoria
                cat_j = items_dict[g_j.item_id].categoria
                if not son_compatibles(cat_i, cat_j):
                    pares_invalidos += 1

    return pares_invalidos / total_pares


def calcular_Dv(
    genes: list[Gen],
    items_dict: dict[str, Item],
    vehiculo: Vehiculo
) -> float:
    """
    Dv — Desperdicio volumetrico.
    Fraccion del volumen total del vehiculo que no esta siendo ocupada.
    """
    volumen_ocupado = 0.0
    for gen in genes:
        item = items_dict[gen.item_id]
        l_ef, a_ef, h_ef = dimensiones_con_amortiguador(gen, item)
        volumen_ocupado += l_ef * a_ef * h_ef

    volumen_total = vehiculo.volumen_total()
    if volumen_total == 0:
        return 0.0

    dv = 1.0 - (volumen_ocupado / volumen_total)
    return max(0.0, dv)


def calcular_Pf(
    genes: list[Gen],
    items_dict: dict[str, Item],
    zonas_dict: dict[str, Zona],
    bboxes: dict[str, tuple],
    trayecto: Trayecto
) -> float:
    """
    Pf — Presion sobre mercancia fragil.
    Para cada item fragil, suma el peso de los items directamente encima.
    Aplica absorcion del amortiguador y factor de vibracion del trayecto.
    Normalizado por el peso total de la carga.
    """
    peso_total = sum(items_dict[g.item_id].peso for g in genes)
    if peso_total == 0:
        return 0.0

    factor_vibracion = 1.0 + trayecto.coef_vibracion  # vibración amplifica presión
    pf_total = 0.0

    for i, gen_fragil in enumerate(genes):
        item_fragil = items_dict[gen_fragil.item_id]
        if item_fragil.fragilidad == 0.0:
            continue

        bb_f = bboxes[gen_fragil.item_id]
        z_techo_fragil = bb_f[5]   # z_max del item fragil

        presion_recibida = 0.0

        for j, gen_encima in enumerate(genes):
            if i == j:
                continue
            bb_e = bboxes[gen_encima.item_id]
            z_piso_encima = bb_e[4]   # z_min del item de arriba

            # Esta encima si su piso toca el techo del fragil (±2cm tolerancia)
            # y se solapa en XY
            toca_z = abs(z_piso_encima - z_techo_fragil) <= 2.0
            solapa_x = bb_f[0] < bb_e[1] and bb_f[1] > bb_e[0]
            solapa_y = bb_f[2] < bb_e[3] and bb_f[3] > bb_e[2]

            if toca_z and solapa_x and solapa_y:
                peso_encima = items_dict[gen_encima.item_id].peso

                # Absorcion del amortiguador del item fragil
                coef_abs = 0.0
                if gen_fragil.tipo_amortiguador != TipoAmortiguador.NINGUNO:
                    from data_model import CaraItem
                    amor = AMORTIGUADORES[gen_fragil.tipo_amortiguador]
                    if CaraItem.TOP in gen_fragil.caras_activas:
                        coef_abs = amor.coef_absorcion

                presion_recibida += peso_encima * (1.0 - coef_abs)

        pf_total += item_fragil.fragilidad * presion_recibida * factor_vibracion

    return pf_total / peso_total


# ─────────────────────────────────────────────
# PENALIZACIONES DURAS
# ─────────────────────────────────────────────

def penalizaciones_duras(
    genes: list[Gen],
    items_dict: dict[str, Item],
    zonas_dict: dict[str, Zona],
    bboxes: dict[str, tuple],
    vehiculo: Vehiculo
) -> float:
    """
    Penaliza violaciones estructurales que el AG debe evitar siempre.
    Retorna un valor de penalizacion acumulada.
    """
    penalizacion = 0.0
    n = len(genes)

    # — Colisiones entre items —
    for i in range(n):
        for j in range(i + 1, n):
            if hay_colision(bboxes[genes[i].item_id], bboxes[genes[j].item_id]):
                penalizacion += PENALIZACION_COLISION

    for gen in genes:
        item  = items_dict[gen.item_id]
        zona  = zonas_dict.get(gen.zona_id)

        if zona is None:
            penalizacion += PENALIZACION_ZONA_PROHIBIDA
            continue

        l_ef, a_ef, h_ef = dimensiones_con_amortiguador(gen, item)

        # — Desborde fuera de los limites de la zona —
        if (gen.x < zona.x_min or gen.x + a_ef > zona.x_max or
            gen.y < zona.y_min or gen.y + l_ef > zona.y_max or
            gen.z < zona.z_min or gen.z + h_ef > zona.z_max):
            penalizacion += PENALIZACION_DESBORDE

        # — Categoria prohibida en la zona —
        if item.categoria in zona.categorias_prohibidas:
            penalizacion += PENALIZACION_ZONA_PROHIBIDA

    # — Peso excedido por zona —
    peso_por_zona = {z.id: 0.0 for z in vehiculo.zonas}
    for gen in genes:
        if gen.zona_id in peso_por_zona:
            peso_por_zona[gen.zona_id] += items_dict[gen.item_id].peso
    for zona in vehiculo.zonas:
        if peso_por_zona[zona.id] > zona.peso_max:
            exceso = (peso_por_zona[zona.id] - zona.peso_max) / zona.peso_max
            penalizacion += PENALIZACION_PESO_ZONA * (1 + exceso)

    # — Peso total excedido —
    peso_total = sum(items_dict[g.item_id].peso for g in genes)
    if peso_total > vehiculo.peso_max_total:
        exceso = (peso_total - vehiculo.peso_max_total) / vehiculo.peso_max_total
        penalizacion += PENALIZACION_PESO_TOTAL * (1 + exceso)

    return penalizacion


# ─────────────────────────────────────────────
# FUNCION DE APTITUD PRINCIPAL
# ─────────────────────────────────────────────

def evaluar(
    cromosoma: Cromosoma,
    items: list[Item],
    vehiculo: Vehiculo,
    trayecto: Trayecto,
    pesos: PesosAptitud = PESOS_DEFAULT
) -> float:
    """
    Calcula y asigna la aptitud del cromosoma.
    Retorna el valor de aptitud (menor es mejor).
    """
    items_dict = {item.id: item for item in items}
    zonas_dict = {zona.id: zona for zona in vehiculo.zonas}

    # Pre-calcular todos los bounding boxes absolutos una sola vez
    bboxes: dict[str, tuple] = {}
    for gen in cromosoma.genes:
        zona = zonas_dict.get(gen.zona_id)
        if zona:
            bboxes[gen.item_id] = bbox_absoluto(gen, items_dict[gen.item_id], zona)
        else:
            # Zona invalida: bbox en el origen como fallback
            bboxes[gen.item_id] = (0, 0, 0, 0, 0, 0)

    # Metricas normalizadas [0, 1]
    dc = calcular_Dc(cromosoma.genes, items_dict, zonas_dict, vehiculo)
    vr = calcular_Vr(cromosoma.genes, items_dict, vehiculo)
    pr = calcular_Pr(cromosoma.genes, items_dict, zonas_dict, bboxes)
    dv = calcular_Dv(cromosoma.genes, items_dict, vehiculo)
    pf = calcular_Pf(cromosoma.genes, items_dict, zonas_dict, bboxes, trayecto)

    aptitud = (
        pesos.w_dc * dc +
        pesos.w_vr * vr +
        pesos.w_pr * pr +
        pesos.w_dv * dv +
        pesos.w_pf * pf +
        penalizaciones_duras(
            cromosoma.genes, items_dict, zonas_dict, bboxes, vehiculo
        )
    )

    cromosoma.aptitud = aptitud
    return aptitud
