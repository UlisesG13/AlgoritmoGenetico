from __future__ import annotations

import math
from typing import Iterable
from io import BytesIO

from data_model import Item
from chromosome import Cromosoma
from fitness import (
    bbox_absoluto, calcular_Dc, calcular_Vr, calcular_Pr, calcular_Dv, calcular_Pf,
    dimensiones_con_amortiguador, son_vecinos, PesosAptitud, PESOS_DEFAULT
)
from knowledge_base import son_compatibles

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


def plot_convergence(historial: Iterable) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib no esta disponible: omitiendo grafica de convergencia.")
        return

    generaciones = [h.generacion for h in historial]
    mejor = [h.mejor_aptitud for h in historial]
    media = [h.media_aptitud for h in historial]
    peor = [h.peor_aptitud for h in historial]

    plt.figure(figsize=(10, 5))
    plt.plot(generaciones, mejor, label="Mejor aptitud", marker="o")
    plt.plot(generaciones, media, label="Media aptitud", marker="x")
    plt.plot(generaciones, peor, label="Peor aptitud", marker="s")
    plt.title("Convergencia del algoritmo genetico")
    plt.xlabel("Generacion")
    plt.ylabel("Aptitud")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def _extraer_bboxes(cromosoma: Cromosoma):
    items = {item.id: item for item in cromosoma.items}
    zonas = {zona.id: zona for zona in cromosoma.vehiculo.zonas}
    bboxes = {}
    for gen in cromosoma.genes:
        item = items[gen.item_id]
        zona = zonas[gen.zona_id]
        bboxes[gen.item_id] = bbox_absoluto(gen, item, zona)
    return bboxes


def _dibujar_caja(ax, bbox, color: str = "tab:blue") -> None:
    x0, x1, y0, y1, z0, z1 = bbox
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        xs = [corners[start][0], corners[end][0]]
        ys = [corners[start][1], corners[end][1]]
        zs = [corners[start][2], corners[end][2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1)


def plot_carga_3d(cromosoma: Cromosoma) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib no esta disponible: omitiendo visualizacion 3D.")
        return

    bboxes = _extraer_bboxes(cromosoma)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colores = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for idx, gen in enumerate(cromosoma.genes):
        bbox = bboxes[gen.item_id]
        color = colores[idx % len(colores)]
        _dibujar_caja(ax, bbox, color=color)
        x_mid = (bbox[0] + bbox[1]) / 2
        y_mid = (bbox[2] + bbox[3]) / 2
        z_mid = (bbox[4] + bbox[5]) / 2
        ax.text(x_mid, y_mid, z_mid, gen.item_id, color="black", fontsize=8)

    ax.set_title("Distribucion de carga 3D")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _calcular_centro_de_masa(cromosoma: Cromosoma) -> tuple[float, float, float]:
    items = {item.id: item for item in cromosoma.items}
    zonas = {zona.id: zona for zona in cromosoma.vehiculo.zonas}
    total_peso = 0.0
    cm_x = cm_y = cm_z = 0.0

    for gen in cromosoma.genes:
        item = items[gen.item_id]
        zona = zonas[gen.zona_id]
        l_ef, a_ef, h_ef = dimensiones_con_amortiguador(gen, item)
        x_abs, y_abs, z_abs = zona.a_absoluto(gen.x, gen.y, gen.z)
        cx = x_abs + a_ef / 2
        cy = y_abs + l_ef / 2
        cz = z_abs + h_ef / 2
        cm_x += item.peso * cx
        cm_y += item.peso * cy
        cm_z += item.peso * cz
        total_peso += item.peso

    if total_peso == 0:
        return (0.0, 0.0, 0.0)

    return (cm_x / total_peso, cm_y / total_peso, cm_z / total_peso)


def plot_center_of_mass(cromosoma: Cromosoma) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib no esta disponible: omitiendo visualizacion del centro de masa.")
        return

    cm = _calcular_centro_de_masa(cromosoma)
    vehiculo = cromosoma.vehiculo

    x_vals = []
    y_vals = []
    vol_total = vehiculo.volumen_total()
    for zona in vehiculo.zonas:
        peso_zona = zona.volumen() / vol_total if vol_total else 0.0
        x_vals.append(zona.x_offset + (zona.x_max - zona.x_min) / 2)
        y_vals.append(zona.y_offset + (zona.y_max - zona.y_min) / 2)

    avg_x = sum(x_vals[i] * (vehiculo.zonas[i].volumen() / vol_total if vol_total else 0.0) for i in range(len(x_vals)))
    avg_y = sum(y_vals[i] * (vehiculo.zonas[i].volumen() / vol_total if vol_total else 0.0) for i in range(len(y_vals)))

    plt.figure(figsize=(7, 7))
    plt.scatter(avg_x, avg_y, label="Centro de estabilidad optimo", color="green", s=100)
    plt.scatter(cm[0], cm[1], label="Centro de masa carga", color="red", s=100)
    plt.title("Centro de masa vs estabilidad optima")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def _calcular_metricas_cromosoma(
    cromosoma: Cromosoma,
    pesos: PesosAptitud = PESOS_DEFAULT
) -> dict[str, float]:
    items = {item.id: item for item in cromosoma.items}
    zonas = {zona.id: zona for zona in cromosoma.vehiculo.zonas}
    bboxes = {}
    for gen in cromosoma.genes:
        item = items[gen.item_id]
        zona = zonas[gen.zona_id]
        bboxes[gen.item_id] = bbox_absoluto(gen, item, zona)

    dc = calcular_Dc(cromosoma.genes, items, zonas, cromosoma.vehiculo)
    vr = calcular_Vr(cromosoma.genes, items, cromosoma.vehiculo)
    pr = calcular_Pr(cromosoma.genes, items, zonas, bboxes)
    dv = calcular_Dv(cromosoma.genes, items, cromosoma.vehiculo)
    pf = calcular_Pf(cromosoma.genes, items, zonas, bboxes, cromosoma.trayecto)

    return {
        "Dc": dc,  # Desplazamiento centro de masa
        "Vr": vr,  # Varianza distribucion peso
        "Pr": pr,  # Penalizacion vecindad
        "Dv": dv,  # Desperdicio volumetrico
        "Pf": pf,  # Presion sobre mercancia fragil
    }


def tabla_comparativa_top3(resultado) -> None:
    print("\n" + "=" * 120)
    print("TABLA COMPARATIVA DE LOS 3 MEJORES INDIVIDUOS".center(120))
    print("=" * 120)

    header = f"{'Rank':<5} {'Aptitud':<12} {'Peso(kg)':<12} {'Dc':<12} {'Vr':<12} {'Pr':<12} {'Dv':<12} {'Pf':<12}"
    print(header)
    print("-" * 120)

    for idx, cromo in enumerate(resultado.top3, start=1):
        metricas = _calcular_metricas_cromosoma(cromo)
        print(
            f"{idx:<5} {cromo.aptitud:<12.4f} {cromo.peso_total():<12.1f} "
            f"{metricas['Dc']:<12.4f} {metricas['Vr']:<12.4f} {metricas['Pr']:<12.4f} "
            f"{metricas['Dv']:<12.4f} {metricas['Pf']:<12.4f}"
        )

    print("=" * 120)
    print("Leyenda: Dc=DesplazamientoCM, Vr=VarianzaPeso, Pr=PenalizacionVecindad, "
          "Dv=DesperdicioVolumetrico, Pf=PresionFragil")
    print()


def resumen_variables_decision(cromosoma: Cromosoma) -> None:
    items = {item.id: item for item in cromosoma.items}
    print("\n" + "=" * 140)
    print("VARIABLES DE DECISIÓN DECODIFICADAS".center(140))
    print("=" * 140)

    header = (
        f"{'Item':<8} {'Zona':<15} {'Pos(cm)':<20} {'Oi(theta_h,theta_v)':<20} "
        f"{'Ai(Amortiguador)':<20} {'Caras':<20} {'Mi(Sujecion)':<15}"
    )
    print(header)
    print("-" * 140)

    for gen in cromosoma.genes:
        item = items[gen.item_id]
        pos_str = f"({gen.x:.0f},{gen.y:.0f},{gen.z:.0f})"
        ori_str = f"({gen.orientacion.theta_h},{gen.orientacion.theta_v})"
        caras_str = ", ".join([c.value for c in gen.caras_activas]) if gen.caras_activas else "ninguna"

        print(
            f"{gen.item_id:<8} {gen.zona_id:<15} {pos_str:<20} {ori_str:<20} "
            f"{gen.tipo_amortiguador.value:<20} {caras_str:<20} {gen.modo_sujecion.value:<15}"
        )

    print("=" * 140)
    print("Variables: Pi=Posicion, Oi=Orientacion, Ai=Amortiguador, Mi=ModoSujecion")
    print()


def reportar_compatibilidad(cromosoma: Cromosoma) -> None:
    items = {item.id: item for item in cromosoma.items}
    zonas = {zona.id: zona for zona in cromosoma.vehiculo.zonas}
    bboxes = _extraer_bboxes(cromosoma)

    print("\n" + "=" * 100)
    print("REPORTE DE COMPATIBILIDAD ENTRE CARGAS (Ri)".center(100))
    print("=" * 100)

    n = len(cromosoma.genes)
    if n < 2:
        print("No hay pares suficientes para evaluar compatibilidad.")
        return

    header = f"{'Item 1':<10} {'Item 2':<10} {'Zona1/Zona2':<20} {'Categoria1':<15} {'Categoria2':<15} {'Estado':<15}"
    print(header)
    print("-" * 100)

    pares_vecinos = 0
    pares_incompatibles = 0

    for i in range(n):
        for j in range(i + 1, n):
            g_i = cromosoma.genes[i]
            g_j = cromosoma.genes[j]
            bb_i = bboxes[g_i.item_id]
            bb_j = bboxes[g_j.item_id]

            if son_vecinos(bb_i, bb_j):
                pares_vecinos += 1
                cat_i = items[g_i.item_id].categoria
                cat_j = items[g_j.item_id].categoria
                compatible = son_compatibles(cat_i, cat_j)
                if not compatible:
                    pares_incompatibles += 1
                estado = "✓ COMPATIBLE" if compatible else "✗ INCOMPATIBLE"

                print(
                    f"{g_i.item_id:<10} {g_j.item_id:<10} {g_i.zona_id}/{g_j.zona_id:<17} "
                    f"{cat_i.value:<15} {cat_j.value:<15} {estado:<15}"
                )

    print("=" * 100)
    print(f"Resumen: {pares_vecinos} pares vecinos detectados, {pares_incompatibles} incompatibles")
    print()


def plot_mapa_sujecion(cromosoma: Cromosoma) -> None:
    """
    Genera un mapa 2D (vista superior) de la plataforma indicando
    qué cargas tienen anclaje y su dirección de sujeción.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib no esta disponible: omitiendo mapa de sujecion.")
        return

    items = {item.id: item for item in cromosoma.items}
    zonas = {zona.id: zona for zona in cromosoma.vehiculo.zonas}
    bboxes = _extraer_bboxes(cromosoma)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Dibujar zonas del vehiculo
    for zona in cromosoma.vehiculo.zonas:
        rect = patches.Rectangle(
            (zona.x_offset, zona.y_offset),
            zona.x_max - zona.x_min,
            zona.y_max - zona.y_min,
            linewidth=2,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.3
        )
        ax.add_patch(rect)
        ax.text(
            zona.x_offset + (zona.x_max - zona.x_min) / 2,
            zona.y_offset + (zona.y_max - zona.y_min) / 2,
            zona.id,
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    # Dibujar items y sus sujeciones
    colores_sujecion = {
        "ninguna": "white",
        "longitudinal": "lightblue",
        "lateral": "lightcoral",
        "completa": "lightgreen",
    }

    for gen in cromosoma.genes:
        bbox = bboxes[gen.item_id]
        x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

        color = colores_sujecion.get(gen.modo_sujecion.value, "white")
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="blue",
            facecolor=color,
            alpha=0.7
        )
        ax.add_patch(rect)

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        ax.text(x_mid, y_mid, gen.item_id, ha="center", va="center", fontweight="bold")

        # Dibujar vectores de sujecion
        if gen.modo_sujecion.value == "longitudinal":
            ax.arrow(x_mid, y_mid, 0, 10, head_width=3, head_length=2, fc="blue", ec="blue")
        elif gen.modo_sujecion.value == "lateral":
            ax.arrow(x_mid, y_mid, 10, 0, head_width=3, head_length=2, fc="red", ec="red")
        elif gen.modo_sujecion.value == "completa":
            ax.arrow(x_mid, y_mid, 8, 8, head_width=3, head_length=2, fc="green", ec="green")

    ax.set_title("Mapa de Sujecion de Cargas (Vista Superior)")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.legend(
        [
            patches.Patch(facecolor="white", edgecolor="blue", label="Sin sujecion"),
            patches.Patch(facecolor="lightblue", edgecolor="blue", label="Sujecion longitudinal (↑)"),
            patches.Patch(facecolor="lightcoral", edgecolor="blue", label="Sujecion lateral (→)"),
            patches.Patch(facecolor="lightgreen", edgecolor="blue", label="Sujecion completa (↗)"),
        ],
        loc="upper right"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def exportar_a_pdf(resultado, nombre_archivo: str = "SAFE_CARGO_reporte.pdf") -> None:
    """
    Exporta un reporte completo a PDF con gráficas y reportes.
    Usa matplotlib para generar PDFs de las gráficas.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib no esta disponible para exportar a PDF")
        return

    # Generar PDF con gráficas
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = nombre_archivo.replace(".pdf", "_graficas.pdf") if nombre_archivo else "SAFE_CARGO_graficas.pdf"

    with PdfPages(pdf_path) as pdf:
        # Página 1: Convergencia
        fig, ax = plt.subplots(figsize=(10, 6))
        generaciones = [h.generacion for h in resultado.historial]
        mejor = [h.mejor_aptitud for h in resultado.historial]
        media = [h.media_aptitud for h in resultado.historial]

        ax.plot(generaciones, mejor, label="Mejor aptitud", marker="o", linewidth=2)
        ax.plot(generaciones, media, label="Media aptitud", marker="x", linewidth=1.5)
        ax.set_title("Convergencia del Algoritmo Genetico", fontsize=14, fontweight="bold")
        ax.set_xlabel("Generacion", fontsize=12)
        ax.set_ylabel("Aptitud", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Página 2: Centro de masa
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = _calcular_centro_de_masa(resultado.mejor_cromosoma)
        vehiculo = resultado.mejor_cromosoma.vehiculo

        x_vals = []
        y_vals = []
        vol_total = vehiculo.volumen_total()
        for zona in vehiculo.zonas:
            x_vals.append(zona.x_offset + (zona.x_max - zona.x_min) / 2)
            y_vals.append(zona.y_offset + (zona.y_max - zona.y_min) / 2)

        avg_x = sum(x_vals[i] * (vehiculo.zonas[i].volumen() / vol_total if vol_total else 0.0) for i in range(len(x_vals)))
        avg_y = sum(y_vals[i] * (vehiculo.zonas[i].volumen() / vol_total if vol_total else 0.0) for i in range(len(y_vals)))

        ax.scatter(avg_x, avg_y, label="Centro de estabilidad optimo", color="green", s=150, marker="*")
        ax.scatter(cm[0], cm[1], label="Centro de masa carga", color="red", s=100, marker="o")
        ax.set_title("Centro de Masa vs Estabilidad Optima", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (cm)", fontsize=12)
        ax.set_ylabel("Y (cm)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_aspect("equal")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Página 3: Mapa de sujeción
        fig, ax = plt.subplots(figsize=(12, 8))
        items = {item.id: item for item in resultado.mejor_cromosoma.items}
        zonas = {zona.id: zona for zona in resultado.mejor_cromosoma.vehiculo.zonas}
        bboxes = _extraer_bboxes(resultado.mejor_cromosoma)

        for zona in resultado.mejor_cromosoma.vehiculo.zonas:
            rect = patches.Rectangle(
                (zona.x_offset, zona.y_offset),
                zona.x_max - zona.x_min,
                zona.y_max - zona.y_min,
                linewidth=2,
                edgecolor="black",
                facecolor="lightgray",
                alpha=0.3
            )
            ax.add_patch(rect)
            ax.text(
                zona.x_offset + (zona.x_max - zona.x_min) / 2,
                zona.y_offset + (zona.y_max - zona.y_min) / 2,
                zona.id,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold"
            )

        colores_sujecion = {"ninguna": "white", "longitudinal": "lightblue", "lateral": "lightcoral", "completa": "lightgreen"}

        for gen in resultado.mejor_cromosoma.genes:
            bbox = bboxes[gen.item_id]
            x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            color = colores_sujecion.get(gen.modo_sujecion.value, "white")
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="blue", facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
            ax.text(x_mid, y_mid, gen.item_id, ha="center", va="center", fontweight="bold")

        ax.set_title("Mapa de Sujecion de Cargas (Vista Superior)", fontsize=14, fontweight="bold")
        ax.set_xlabel("X (cm)", fontsize=12)
        ax.set_ylabel("Y (cm)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_aspect("equal")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    txt_path = nombre_archivo

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("SAFE-CARGO: INFORME DE OPTIMIZACIÓN\n")
        f.write("=" * 120 + "\n\n")

        f.write("1. TABLA COMPARATIVA DE LOS 3 MEJORES INDIVIDUOS\n")
        f.write("-" * 120 + "\n")
        header = f"{'Rank':<5} {'Aptitud':<12} {'Peso(kg)':<12} {'Dc':<12} {'Vr':<12} {'Pr':<12} {'Dv':<12} {'Pf':<12}\n"
        f.write(header)
        f.write("-" * 120 + "\n")

        for idx, cromo in enumerate(resultado.top3, start=1):
            metricas = _calcular_metricas_cromosoma(cromo)
            row = (
                f"{idx:<5} {cromo.aptitud:<12.4f} {cromo.peso_total():<12.1f} "
                f"{metricas['Dc']:<12.4f} {metricas['Vr']:<12.4f} {metricas['Pr']:<12.4f} "
                f"{metricas['Dv']:<12.4f} {metricas['Pf']:<12.4f}\n"
            )
            f.write(row)

        f.write("\n\n2. VARIABLES DE DECISIÓN (MEJOR SOLUCIÓN)\n")
        f.write("-" * 140 + "\n")
        f.write(f"{'Item':<8} {'Zona':<15} {'Pos(cm)':<20} {'Oi(θh,θv)':<20} "
                f"{'Ai(Amortiguador)':<20} {'Mi(Sujeción)':<15}\n")
        f.write("-" * 140 + "\n")

        items = {item.id: item for item in resultado.mejor_cromosoma.items}
        for gen in resultado.mejor_cromosoma.genes:
            pos_str = f"({gen.x:.1f},{gen.y:.1f},{gen.z:.1f})"
            ori_str = f"({gen.orientacion.theta_h},{gen.orientacion.theta_v})"
            f.write(f"{gen.item_id:<8} {gen.zona_id:<15} {pos_str:<20} {ori_str:<20} "
                    f"{gen.tipo_amortiguador.value:<20} {gen.modo_sujecion.value:<15}\n")

        f.write("\n\n3. REPORTE DE COMPATIBILIDAD DE VECINDAD\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Item 1':<10} {'Item 2':<10} {'Zonas':<20} {'Categoría 1':<15} {'Categoría 2':<15} {'Estado':<15}\n")
        f.write("-" * 100 + "\n")

        bboxes = _extraer_bboxes(resultado.mejor_cromosoma)
        n = len(resultado.mejor_cromosoma.genes)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                g_i = resultado.mejor_cromosoma.genes[i]
                g_j = resultado.mejor_cromosoma.genes[j]
                bb_i = bboxes[g_i.item_id]
                bb_j = bboxes[g_j.item_id]

                if son_vecinos(bb_i, bb_j):
                    count += 1
                    cat_i = items[g_i.item_id].categoria
                    cat_j = items[g_j.item_id].categoria
                    compatible = son_compatibles(cat_i, cat_j)
                    estado = "✓ COMPATIBLE" if compatible else "✗ INCOMP."

                    f.write(f"{g_i.item_id:<10} {g_j.item_id:<10} {g_i.zona_id}/{g_j.zona_id:<18} "
                            f"{cat_i.value:<15} {cat_j.value:<15} {estado:<15}\n")

        if count == 0:
            f.write("No hay pares de items vecinos detectados.\n")

        f.write("\n\n4. ESTADÍSTICAS GENERALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"Generaciones ejecutadas: {resultado.generaciones}\n")
        f.write(f"Tiempo total (segundos): {resultado.tiempo_total_seg:.2f}\n")
        f.write(f"Convergencia anticipada: {'Sí' if resultado.convergencia else 'No'}\n")
        f.write(f"Mejor aptitud encontrada: {resultado.mejor_cromosoma.aptitud:.6f}\n")
        f.write(f"Peso total de carga: {resultado.mejor_cromosoma.peso_total():.1f} kg\n")
        f.write(f"\nPDF con gráficas: {pdf_path}\n")

    print(f"\n✓ Reporte en texto generado: {txt_path}")
    print(f"✓ Gráficas exportadas a PDF: {pdf_path}")

