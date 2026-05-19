"""
SAFE-CARGO | main.py
Punto de entrada para ejecutar el algoritmo genético y generar reportes.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from knowledge_base import CATALOGO_ITEMS, TRAYECTOS, crear_vehiculo_prueba, item_por_id
from genetic_engine import ParametrosAG, ejecutar
from visualization import (
    plot_convergence,
    plot_center_of_mass,
    plot_carga_3d,
    plot_mapa_sujecion,
    tabla_comparativa_top3,
    resumen_variables_decision,
    reportar_compatibilidad,
    exportar_a_pdf,
)


def seleccionar_items(ids: Sequence[str], count: int) -> list:
    if ids:
        seleccion = []
        for item_id in ids:
            item = item_por_id(item_id)
            if item is None:
                raise ValueError(f"Item desconocido: {item_id}")
            seleccion.append(item)
        return seleccion

    if count <= 0:
        raise ValueError("El numero de items debe ser mayor que cero.")

    return CATALOGO_ITEMS[:count]


def imprimir_individuo(cromosoma, idx: int) -> None:
    print(f"\n--- Solucion #{idx} ---")
    print(f"Aptitud: {cromosoma.aptitud:.4f}")
    print(f"Peso total: {cromosoma.peso_total():.1f} kg")
    for gen in cromosoma.genes:
        print(
            f"  Item={gen.item_id} | Zona={gen.zona_id} | "
            f"Pos=({gen.x:.1f},{gen.y:.1f},{gen.z:.1f})cm | "
            f"Ori=({gen.orientacion.theta_h},{gen.orientacion.theta_v}) | "
            f"Amort={gen.tipo_amortiguador.value} | "
            f"Sujecion={gen.modo_sujecion.value}"
        )


def imprimir_resumen(resultado) -> None:
    print("=== SAFE-CARGO | Resumen de ejecucion ===")
    print(f"Generaciones ejecutadas : {resultado.generaciones}")
    print(f"Tiempo total (s)         : {resultado.tiempo_total_seg:.2f}")
    print(f"Convergencia anticipada  : {resultado.convergencia}")
    print(f"Mejor aptitud encontrada : {resultado.mejor_cromosoma.aptitud:.4f}")
    for idx, cromo in enumerate(resultado.top3, start=1):
        imprimir_individuo(cromo, idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el algoritmo genetico SAFE-CARGO para optimizar el empaquetado." 
    )
    parser.add_argument(
        "--trayecto",
        choices=list(TRAYECTOS.keys()),
        default="moderado",
        help="Escenario de trayecto a usar.",
    )
    parser.add_argument(
        "--items",
        help="Lista separada por comas de ids de items a transportar.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="Numero de items del catalogo a usar si no se especifican ids.",
    )
    parser.add_argument(
        "--tam-poblacion",
        type=int,
        default=60,
        help="Tamano de la poblacion del algoritmo genetico.",
    )
    parser.add_argument(
        "--max-generaciones",
        type=int,
        default=150,
        help="Numero maximo de generaciones a ejecutar.",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--tipo-cruce",
        choices=["uniforme", "un_punto"],
        default="uniforme",
        help="Tipo de operador de cruce.",
    )
    parser.add_argument(
        "--visualizar",
        action="store_true",
        help="Genera graficas de convergencia e informacion de la mejor solucion.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Exporta el reporte completo a archivo PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehiculo = crear_vehiculo_prueba()
    trayecto = TRAYECTOS[args.trayecto]

    items_ids = [item_id.strip() for item_id in args.items.split(",")] if args.items else []
    items = seleccionar_items(items_ids, args.count)

    params = ParametrosAG(
        tam_poblacion=args.tam_poblacion,
        max_generaciones=args.max_generaciones,
        tipo_cruce=args.tipo_cruce,
        semilla=args.semilla,
    )

    resultado = ejecutar(items, vehiculo, trayecto, params)
    imprimir_resumen(resultado)

    # Salidas esperadas del sistema
    print("\n" + "=" * 120)
    print("SALIDAS DEL SISTEMA".center(120))
    print("=" * 120)

    tabla_comparativa_top3(resultado)
    resumen_variables_decision(resultado.mejor_cromosoma)
    reportar_compatibilidad(resultado.mejor_cromosoma)

    if args.pdf:
        exportar_a_pdf(resultado, "SAFE_CARGO_reporte.pdf")

    if args.visualizar:
        print("\nGenerando visualizaciones graficas...")
        plot_convergence(resultado.historial)
        plot_center_of_mass(resultado.mejor_cromosoma)
        plot_carga_3d(resultado.mejor_cromosoma)
        plot_mapa_sujecion(resultado.mejor_cromosoma)


if __name__ == "__main__":
    main()
