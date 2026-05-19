"""
SAFE-CARGO | genetic_engine.py
Motor principal del algoritmo genetico.

Orquesta el ciclo evolutivo completo:
    1. Inicializacion de la poblacion
    2. Evaluacion de aptitud
    3. Seleccion de padres
    4. Cruce y generacion de hijos
    5. Mutacion
    6. Elitismo y reemplazo
    7. Registro de estadisticas por generacion

La configuracion del AG se centraliza en ParametrosAG.
Los resultados se devuelven en un objeto ResultadoAG que
visualization.py y main.py consumen directamente.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from data_model import Item, Vehiculo, Trayecto
from chromosome import Cromosoma, crear_poblacion
from fitness import evaluar, PesosAptitud, PESOS_DEFAULT
from operators import (
    seleccion_torneo,
    seleccion_ruleta,
    cruce_uniforme,
    cruce_un_punto,
    mutar_cromosoma,
    aplicar_elitismo,
    PROB_MUTAR_ZONA,
    PROB_MUTAR_POSICION,
    PROB_MUTAR_ORIENTACION,
    PROB_MUTAR_AMORTIGUADOR,
    PROB_MUTAR_SUJECION,
)


# ─────────────────────────────────────────────
# PARAMETROS DEL AG
# ─────────────────────────────────────────────

@dataclass
class ParametrosAG:
    """
    Configuracion completa del algoritmo genetico.

    Todos los valores tienen defaults razonables para SAFE-CARGO.
    Se pueden sobreescribir al instanciar o modificar antes de
    llamar a ejecutar().

    Atributos
    ─────────
    tam_poblacion     : numero de individuos por generacion.
    max_generaciones  : limite de iteraciones evolutivas.
    prob_cruce        : probabilidad de cruzar dos padres (vs clonar directamente).
    tipo_cruce        : 'uniforme' o 'un_punto'.
    tam_torneo        : k participantes en seleccion por torneo.
    usar_ruleta       : True = ruleta, False = torneo (default).
    n_elite           : individuos elite que pasan directamente a la sig. generacion.
    prob_mut_zona     : prob. de mutar la zona de cada gen.
    prob_mut_posicion : prob. de perturbar x/y/z de cada gen.
    prob_mut_ori      : prob. de mutar orientacion de cada gen.
    prob_mut_amort    : prob. de mutar amortiguador de cada gen.
    prob_mut_sujecion : prob. de mutar modo de sujecion de cada gen.
    semilla           : semilla aleatoria para reproducibilidad (None = aleatoria).
    tolerancia        : mejora minima entre generaciones para no contar como estancamiento.
    paciencia         : generaciones sin mejora antes de detener (0 = sin parada temprana).
    pesos_aptitud     : instancia de PesosAptitud para la funcion objetivo.
    callback          : funcion opcional llamada al final de cada generacion.
                        Firma: callback(generacion: int, stats: EstadisticaGen) -> None
    """
    tam_poblacion     : int            = 60
    max_generaciones  : int            = 150
    prob_cruce        : float          = 0.85
    tipo_cruce        : str            = "uniforme"    # "uniforme" | "un_punto"
    tam_torneo        : int            = 3
    usar_ruleta       : bool           = False
    n_elite           : int            = 2
    prob_mut_zona     : float          = PROB_MUTAR_ZONA
    prob_mut_posicion : float          = PROB_MUTAR_POSICION
    prob_mut_ori      : float          = PROB_MUTAR_ORIENTACION
    prob_mut_amort    : float          = PROB_MUTAR_AMORTIGUADOR
    prob_mut_sujecion : float          = PROB_MUTAR_SUJECION
    semilla           : int | None     = None
    tolerancia        : float          = 1e-6
    paciencia         : int            = 30            # 0 = desactivado
    pesos_aptitud     : PesosAptitud   = field(default_factory=PesosAptitud)
    callback          : Callable | None = field(default=None, repr=False)


# ─────────────────────────────────────────────
# ESTADISTICAS POR GENERACION
# ─────────────────────────────────────────────

@dataclass
class EstadisticaGen:
    """
    Metricas registradas al finalizar cada generacion.
    visualization.py lee la lista de estas para graficar la convergencia.
    """
    generacion    : int
    mejor_aptitud : float
    media_aptitud : float
    peor_aptitud  : float
    tiempo_seg    : float          # tiempo acumulado desde inicio


# ─────────────────────────────────────────────
# RESULTADO FINAL DEL AG
# ─────────────────────────────────────────────

@dataclass
class ResultadoAG:
    """
    Objeto retornado por ejecutar() con todo lo necesario para
    reportes y visualizaciones.

    Atributos
    ─────────
    mejor_cromosoma  : solucion con menor aptitud encontrada.
    top3             : los 3 mejores individuos unicos al finalizar.
    historial        : lista de EstadisticaGen, una por generacion.
    generaciones     : numero de generaciones realmente ejecutadas.
    tiempo_total_seg : tiempo de ejecucion total en segundos.
    convergencia     : True si se detecto estancamiento antes de max_generaciones.
    parametros       : configuracion usada en esta ejecucion.
    """
    mejor_cromosoma  : Cromosoma
    top3             : list[Cromosoma]
    historial        : list[EstadisticaGen]
    generaciones     : int
    tiempo_total_seg : float
    convergencia     : bool
    parametros       : ParametrosAG


# ─────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────

def _evaluar_poblacion(
    poblacion     : list[Cromosoma],
    items         : list[Item],
    vehiculo      : Vehiculo,
    trayecto      : Trayecto,
    pesos         : PesosAptitud,
) -> None:
    """Evalua in-place solo los cromosomas sin aptitud asignada."""
    for cromo in poblacion:
        if cromo.aptitud is None:
            evaluar(cromo, items, vehiculo, trayecto, pesos)


def _estadisticas(
    poblacion  : list[Cromosoma],
    generacion : int,
    t_inicio   : float,
) -> EstadisticaGen:
    """Calcula estadisticas de la generacion actual."""
    aptitudes = [c.aptitud for c in poblacion if c.aptitud is not None]
    return EstadisticaGen(
        generacion    = generacion,
        mejor_aptitud = min(aptitudes),
        media_aptitud = sum(aptitudes) / len(aptitudes),
        peor_aptitud  = max(aptitudes),
        tiempo_seg    = time.time() - t_inicio,
    )


def _seleccionar_padre(
    poblacion   : list[Cromosoma],
    params      : ParametrosAG,
) -> Cromosoma:
    """Selecciona un padre segun el metodo configurado."""
    if params.usar_ruleta:
        return seleccion_ruleta(poblacion)
    return seleccion_torneo(poblacion, k=params.tam_torneo)


def _cruzar(
    padre1 : Cromosoma,
    padre2 : Cromosoma,
    params : ParametrosAG,
) -> tuple[Cromosoma, Cromosoma]:
    """Aplica el operador de cruce configurado."""
    if params.tipo_cruce == "un_punto":
        return cruce_un_punto(padre1, padre2)
    return cruce_uniforme(padre1, padre2)


def _top_n(poblacion: list[Cromosoma], n: int) -> list[Cromosoma]:
    """Retorna los n mejores cromosomas ordenados por aptitud ascendente."""
    evaluados = [c for c in poblacion if c.aptitud is not None]
    return sorted(evaluados, key=lambda c: c.aptitud)[:n]


# ─────────────────────────────────────────────
# MOTOR PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar(
    items    : list[Item],
    vehiculo : Vehiculo,
    trayecto : Trayecto,
    params   : ParametrosAG | None = None,
) -> ResultadoAG:
    """
    Ejecuta el algoritmo genetico completo y retorna el ResultadoAG.

    Ciclo por generacion:
        1. Evaluar cromosomas sin aptitud
        2. Registrar estadisticas
        3. Llamar callback (si existe)
        4. Verificar parada temprana
        5. Generar nueva poblacion mediante seleccion + cruce
        6. Mutar la nueva poblacion
        7. Aplicar elitismo
        8. Reemplazar poblacion

    Args:
        items    : lista de items a transportar.
        vehiculo : vehiculo de transporte.
        trayecto : condiciones topograficas del viaje.
        params   : configuracion del AG (usa defaults si es None).

    Returns:
        ResultadoAG con la mejor solucion, historial y metricas.
    """
    if params is None:
        params = ParametrosAG()

    # Semilla aleatoria para reproducibilidad
    if params.semilla is not None:
        random.seed(params.semilla)

    t_inicio = time.time()
    historial: list[EstadisticaGen] = []

    # ── 1. Poblacion inicial ──────────────────────────────────────────
    poblacion = crear_poblacion(params.tam_poblacion, items, vehiculo, trayecto)
    _evaluar_poblacion(poblacion, items, vehiculo, trayecto, params.pesos_aptitud)

    mejor_aptitud_global = min(c.aptitud for c in poblacion)
    generaciones_sin_mejora = 0
    convergencia = False

    # ── Ciclo evolutivo ───────────────────────────────────────────────
    for gen_actual in range(1, params.max_generaciones + 1):

        # Estadisticas de la generacion actual
        stats = _estadisticas(poblacion, gen_actual, t_inicio)
        historial.append(stats)

        # Callback externo (para progress bars, logs, etc.)
        if params.callback is not None:
            params.callback(gen_actual, stats)

        # Parada temprana por convergencia
        if params.paciencia > 0:
            mejora = mejor_aptitud_global - stats.mejor_aptitud
            if mejora > params.tolerancia:
                mejor_aptitud_global = stats.mejor_aptitud
                generaciones_sin_mejora = 0
            else:
                generaciones_sin_mejora += 1

            if generaciones_sin_mejora >= params.paciencia:
                convergencia = True
                break

        # ── 2. Generar nueva poblacion ────────────────────────────────
        nueva_poblacion: list[Cromosoma] = []

        # Necesitamos tam_poblacion hijos; cada cruce produce 2
        while len(nueva_poblacion) < params.tam_poblacion:
            padre1 = _seleccionar_padre(poblacion, params)
            padre2 = _seleccionar_padre(poblacion, params)

            if random.random() < params.prob_cruce:
                hijo1, hijo2 = _cruzar(padre1, padre2, params)
            else:
                # Sin cruce: clonar directamente (aptitud invalidada)
                hijo1 = padre1.clonar()
                hijo2 = padre2.clonar()
                hijo1.aptitud = None
                hijo2.aptitud = None

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < params.tam_poblacion:
                nueva_poblacion.append(hijo2)

        # ── 3. Mutacion ───────────────────────────────────────────────
        for cromo in nueva_poblacion:
            mutar_cromosoma(
                cromo,
                prob_zona        = params.prob_mut_zona,
                prob_posicion    = params.prob_mut_posicion,
                prob_orientacion = params.prob_mut_ori,
                prob_amort       = params.prob_mut_amort,
                prob_sujecion    = params.prob_mut_sujecion,
            )

        # ── 4. Evaluar hijos ──────────────────────────────────────────
        _evaluar_poblacion(nueva_poblacion, items, vehiculo, trayecto, params.pesos_aptitud)

        # ── 5. Elitismo + reemplazo ───────────────────────────────────
        poblacion = aplicar_elitismo(poblacion, nueva_poblacion, n_elite=params.n_elite)

    # ── Resultado final ───────────────────────────────────────────────
    top3 = _top_n(poblacion, 3)

    return ResultadoAG(
        mejor_cromosoma  = top3[0].clonar(),
        top3             = [c.clonar() for c in top3],
        historial        = historial,
        generaciones     = len(historial),
        tiempo_total_seg = time.time() - t_inicio,
        convergencia     = convergencia,
        parametros       = params,
    )


# ─────────────────────────────────────────────
# BLOQUE DE PRUEBA RAPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from knowledge_base import CATALOGO_ITEMS, TRAYECTOS, crear_vehiculo_prueba

    vehiculo     = crear_vehiculo_prueba()
    trayecto     = TRAYECTOS["moderado"]
    items_prueba = CATALOGO_ITEMS[:6]

    print("=== SAFE-CARGO | genetic_engine.py — prueba ===\n")
    print(f"Items     : {[i.id for i in items_prueba]}")
    print(f"Vehiculo  : {vehiculo.nombre}")
    print(f"Trayecto  : {trayecto.nombre}\n")

    # Configuracion reducida para prueba rapida
    params = ParametrosAG(
        tam_poblacion    = 30,
        max_generaciones = 50,
        prob_cruce       = 0.85,
        tipo_cruce       = "uniforme",
        tam_torneo       = 3,
        n_elite          = 2,
        paciencia        = 15,
        semilla          = 42,
    )

    # Callback simple para ver progreso
    def log_gen(generacion: int, stats: EstadisticaGen) -> None:
        if generacion % 10 == 0 or generacion == 1:
            print(f"  Gen {generacion:>3} | "
                  f"mejor={stats.mejor_aptitud:.4f}  "
                  f"media={stats.media_aptitud:.4f}  "
                  f"peor={stats.peor_aptitud:.4f}  "
                  f"t={stats.tiempo_seg:.2f}s")

    params.callback = log_gen

    print("Ejecutando AG...\n")
    resultado = ejecutar(items_prueba, vehiculo, trayecto, params)

    print(f"\n{'='*50}")
    print(f"RESULTADO FINAL")
    print(f"{'='*50}")
    print(f"Generaciones ejecutadas : {resultado.generaciones}")
    print(f"Tiempo total            : {resultado.tiempo_total_seg:.2f}s")
    print(f"Convergencia anticipada : {resultado.convergencia}")
    print(f"Mejor aptitud           : {resultado.mejor_cromosoma.aptitud:.6f}")

    print(f"\nTOP 3 soluciones:")
    for i, cromo in enumerate(resultado.top3, 1):
        print(f"  #{i} aptitud={cromo.aptitud:.6f}  peso={cromo.peso_total():.1f}kg")
        for gen in cromo.genes:
            print(f"      {gen}")

    print(f"\nHistorial (primeras y ultimas 3 generaciones):")
    for s in resultado.historial[:3]:
        print(f"  Gen {s.generacion:>3}: mejor={s.mejor_aptitud:.4f}")
    print("  ...")
    for s in resultado.historial[-3:]:
        print(f"  Gen {s.generacion:>3}: mejor={s.mejor_aptitud:.4f}")

    # Verificaciones basicas
    assert resultado.mejor_cromosoma.aptitud is not None
    assert len(resultado.top3) <= 3
    assert len(resultado.historial) == resultado.generaciones
    assert resultado.tiempo_total_seg > 0
    print("\n✓ Todas las verificaciones pasaron")
    print("=== Prueba completada ===")
