"""
SAFE-CARGO | chromosome.py
Representacion genetica de una solucion de empaquetado.

Cada individuo (Cromosoma) es una lista de Genes, uno por cada item
a transportar. El gen codifica todas las variables de decision:
    - Pi  : zona asignada + posicion LOCAL dentro de ella (x, y, z en cm)
    - Oi  : orientacion discreta (theta_h, theta_v)
    - Ai  : tipo de amortiguador + caras activas
    - Mi  : modo de sujecion

SISTEMA DE COORDENADAS
──────────────────────
Las posiciones (x, y, z) son LOCALES a la zona asignada, en cm:
    x : eje ancho, en [zona.x_min, zona.x_max]
    y : eje largo, en [zona.y_min, zona.y_max]
    z : eje alto,  en [zona.z_min, zona.z_max]

fitness.py compara gen.x directamente contra zona.x_min/x_max y llama
a zona.a_absoluto(gen.x, gen.y, gen.z) para convertir a coordenadas
globales del vehiculo.

PENDIENTE EN data_model.py (agregar a clase Zona)
──────────────────────────────────────────────────
    x_offset : float = 0.0   # coord X global de la esquina inf-izq de la zona
    y_offset : float = 0.0   # coord Y global
    z_offset : float = 0.0   # coord Z global

    def a_absoluto(self, x_local: float, y_local: float, z_local: float
                   ) -> tuple[float, float, float]:
        return (x_local + self.x_offset,
                y_local + self.y_offset,
                z_local + self.z_offset)
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field

from data_model import (
    Item, Vehiculo, Trayecto,
    Orientacion, Amortiguador, AMORTIGUADORES,
    CaraItem, TipoAmortiguador, ModoSujecion, Zona,
)


# ─────────────────────────────────────────────
# GEN — variables de decision para un solo item
# ─────────────────────────────────────────────

@dataclass
class Gen:
    """
    Encapsula todas las decisiones de colocacion para un Item concreto.

    Atributos
    ─────────
    item_id           : ID del item que representa este gen.
    zona_id           : ID de la zona asignada (Zi).
    x                 : posicion local en X dentro de la zona [cm].
    y                 : posicion local en Y dentro de la zona [cm].
    z                 : posicion local en Z dentro de la zona [cm].
    orientacion       : rotacion discreta theta_h y theta_v (Oi).
    tipo_amortiguador : material de buffer seleccionado (Ai).
    caras_activas     : caras del item que reciben el buffer (Ai).
    modo_sujecion     : tipo de anclaje asignado (Mi).

    Convenio de posicion
    ────────────────────
    (x, y, z) es la esquina INFERIOR-FRONTAL-IZQUIERDA del item
    en el espacio LOCAL de la zona. Se compara directamente contra
    zona.x_min / zona.x_max (usados en fitness.py para deteccion
    de desborde).
    """
    item_id           : str
    zona_id           : str
    x                 : float          # cm — local a la zona
    y                 : float          # cm — local a la zona
    z                 : float          # cm — local a la zona
    orientacion       : Orientacion
    tipo_amortiguador : TipoAmortiguador
    caras_activas     : list[CaraItem] = field(default_factory=list)
    modo_sujecion     : ModoSujecion   = ModoSujecion.NINGUNA

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def amortiguador_efectivo(self) -> Amortiguador:
        """
        Retorna una copia del amortiguador base con las caras_activas
        de este gen aplicadas. El mismo tipo de buffer puede cubrir
        distintas caras en distintos items.
        """
        base = AMORTIGUADORES[self.tipo_amortiguador]
        resultado = deepcopy(base)
        resultado.caras_activas = list(self.caras_activas)
        return resultado

    def clonar(self) -> Gen:
        return deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"Gen(item={self.item_id!r}, zona={self.zona_id!r}, "
            f"pos=({self.x:.1f},{self.y:.1f},{self.z:.1f})cm, "
            f"ori=({self.orientacion.theta_h}h,{self.orientacion.theta_v}v), "
            f"buf={self.tipo_amortiguador.value}, suj={self.modo_sujecion.value})"
        )


# ─────────────────────────────────────────────
# CROMOSOMA — individuo completo del AG
# ─────────────────────────────────────────────

@dataclass
class Cromosoma:
    """
    Individuo del algoritmo genetico.

    Contiene un Gen por cada item de la lista `items`. El orden en
    `genes` coincide con el orden en `items` para acceso directo por
    indice.

    Atributos
    ─────────
    genes    : lista de genes (uno por item).
    items    : referencia compartida a los items a transportar.
    vehiculo : referencia compartida al vehiculo de transporte.
    trayecto : referencia compartida a las condiciones topograficas.
    aptitud  : valor calculado por fitness.py (None = no evaluado).
    """
    genes    : list[Gen]
    items    : list[Item]
    vehiculo : Vehiculo
    trayecto : Trayecto
    aptitud  : float | None = field(default=None, compare=False)

    # ------------------------------------------------------------------
    # Constructor aleatorio
    # ------------------------------------------------------------------

    @classmethod
    def aleatorio(
        cls,
        items    : list[Item],
        vehiculo : Vehiculo,
        trayecto : Trayecto,
    ) -> Cromosoma:
        """
        Genera un Cromosoma con genes completamente aleatorios.

        La zona se elige al azar sin filtrar incompatibilidades.
        La funcion de aptitud penalizara asignaciones invalidas;
        esto preserva diversidad genetica en la poblacion inicial.

        Las coordenadas x/y/z se generan uniformemente dentro de
        los limites [x_min, x_max] de la zona sorteada.
        """
        genes: list[Gen] = []

        prob_sujecion = min(
            (trayecto.inclinacion_long + trayecto.inclinacion_lat) / 40.0,
            1.0
        )

        for item in items:
            zona = random.choice(vehiculo.zonas)

            # Posicion local en cm dentro de los limites de la zona
            x = random.uniform(zona.x_min, zona.x_max)
            y = random.uniform(zona.y_min, zona.y_max)
            z = random.uniform(zona.z_min, zona.z_max)

            # Orientacion discreta
            orientacion = Orientacion(
                theta_h = random.choice([0, 90, 180, 270]),
                theta_v = random.choice([0, 90]),
            )

            # Amortiguador sesgado por fragilidad del item
            if random.random() < item.fragilidad:
                tipo_amort = random.choice([
                    TipoAmortiguador.CARTON,
                    TipoAmortiguador.PLASTICO_BURBUJA,
                    TipoAmortiguador.ALGODON,
                ])
                n_caras = random.randint(1, 4)
                caras   = random.sample(list(CaraItem), n_caras)
            else:
                tipo_amort = TipoAmortiguador.NINGUNO
                caras      = []

            # Modo de sujecion sesgado por severidad del trayecto
            if random.random() < prob_sujecion:
                modo = random.choice([
                    ModoSujecion.LONGITUDINAL,
                    ModoSujecion.LATERAL,
                    ModoSujecion.COMPLETA,
                ])
            else:
                modo = ModoSujecion.NINGUNA

            genes.append(Gen(
                item_id           = item.id,
                zona_id           = zona.id,
                x                 = x,
                y                 = y,
                z                 = z,
                orientacion       = orientacion,
                tipo_amortiguador = tipo_amort,
                caras_activas     = caras,
                modo_sujecion     = modo,
            ))

        return cls(genes=genes, items=items, vehiculo=vehiculo, trayecto=trayecto)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def gen_de_item(self, item_id: str) -> Gen | None:
        """Retorna el gen correspondiente a un item_id dado."""
        for gen in self.genes:
            if gen.item_id == item_id:
                return gen
        return None

    def items_en_zona(self, zona_id: str) -> list[tuple[Item, Gen]]:
        """Retorna lista de (Item, Gen) para todos los items en una zona."""
        item_map = {it.id: it for it in self.items}
        return [
            (item_map[g.item_id], g)
            for g in self.genes
            if g.zona_id == zona_id and g.item_id in item_map
        ]

    def peso_total(self) -> float:
        """Suma de masa de todos los items."""
        item_map = {it.id: it for it in self.items}
        return sum(item_map[g.item_id].peso for g in self.genes if g.item_id in item_map)

    def peso_en_zona(self, zona_id: str) -> float:
        """Suma de masa de los items asignados a una zona."""
        item_map = {it.id: it for it in self.items}
        return sum(
            item_map[g.item_id].peso
            for g in self.genes
            if g.zona_id == zona_id and g.item_id in item_map
        )

    def zona_de_gen(self, gen: Gen) -> Zona | None:
        """Retorna el objeto Zona del vehiculo para un gen dado."""
        return self.vehiculo.zona_por_id(gen.zona_id)

    def clonar(self) -> Cromosoma:
        """
        Copia profunda del cromosoma.
        items, vehiculo y trayecto son referencias compartidas
        (son de solo lectura durante el AG).
        """
        return Cromosoma(
            genes    = [g.clonar() for g in self.genes],
            items    = self.items,
            vehiculo = self.vehiculo,
            trayecto = self.trayecto,
            aptitud  = self.aptitud,
        )

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        apt_str = f"{self.aptitud:.4f}" if self.aptitud is not None else "sin evaluar"
        return f"Cromosoma(items={len(self.genes)}, aptitud={apt_str})"


# ─────────────────────────────────────────────
# POBLACION — coleccion de cromosomas
# ─────────────────────────────────────────────

def crear_poblacion(
    n        : int,
    items    : list[Item],
    vehiculo : Vehiculo,
    trayecto : Trayecto,
) -> list[Cromosoma]:
    """
    Genera una poblacion inicial de n cromosomas aleatorios.

    Args:
        n        : tamanio de la poblacion.
        items    : lista de items a transportar.
        vehiculo : vehiculo de transporte.
        trayecto : condiciones topograficas del viaje.

    Returns:
        Lista de n Cromosomas sin evaluar (aptitud=None).
    """
    return [Cromosoma.aleatorio(items, vehiculo, trayecto) for _ in range(n)]


# ─────────────────────────────────────────────
# BLOQUE DE PRUEBA RAPIDA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from knowledge_base import CATALOGO_ITEMS, TRAYECTOS, crear_vehiculo_prueba

    vehiculo     = crear_vehiculo_prueba()
    trayecto     = TRAYECTOS["severo"]
    items_prueba = CATALOGO_ITEMS[:5]

    print("=== SAFE-CARGO | chromosome.py — prueba ===\n")
    print(f"Items      : {[i.id for i in items_prueba]}")
    print(f"Vehiculo   : {vehiculo.nombre}")
    print(f"Trayecto   : {trayecto.nombre}\n")

    # Crear cromosoma individual
    c = Cromosoma.aleatorio(items_prueba, vehiculo, trayecto)
    print(f"Cromosoma  : {c}")
    print(f"Peso total : {c.peso_total():.1f} kg\n")

    print("Genes:")
    for gen in c.genes:
        zona = vehiculo.zona_por_id(gen.zona_id)
        print(f"  {gen}")
        dentro_x = zona.x_min <= gen.x <= zona.x_max
        dentro_y = zona.y_min <= gen.y <= zona.y_max
        dentro_z = zona.z_min <= gen.z <= zona.z_max
        print(f"    en limites zona: X={dentro_x} Y={dentro_y} Z={dentro_z}")

    print("\nItems en Z_BATEA:")
    for item, gen in c.items_en_zona("Z_BATEA"):
        print(f"  {item.id} ({item.nombre})")
        print(f"    pos=({gen.x:.1f},{gen.y:.1f},{gen.z:.1f})cm "
              f"buf={gen.tipo_amortiguador.value} suj={gen.modo_sujecion.value}")

    # Crear y verificar poblacion
    print(f"\nGenerando poblacion de 10 cromosomas...")
    poblacion = crear_poblacion(10, items_prueba, vehiculo, trayecto)
    print(f"Poblacion  : {len(poblacion)} individuos")

    # Verificar clonar
    original = poblacion[0]
    clon = original.clonar()
    clon.genes[0].zona_id = "TEST"
    assert original.genes[0].zona_id != "TEST", "ERROR: clonar() comparte referencia"
    print("✓ clonar() genera copia profunda independiente")

    # Verificar coordenadas dentro de limites
    zonas_dict = {z.id: z for z in vehiculo.zonas}
    for cromo in poblacion:
        for gen in cromo.genes:
            z = zonas_dict[gen.zona_id]
            assert z.x_min <= gen.x <= z.x_max, f"X fuera de rango en {gen.item_id}"
            assert z.y_min <= gen.y <= z.y_max, f"Y fuera de rango en {gen.item_id}"
            assert z.z_min <= gen.z <= z.z_max, f"Z fuera de rango en {gen.item_id}"
    print("✓ Todas las coordenadas dentro de limites de zona")
    print("\n=== Prueba completada ===")