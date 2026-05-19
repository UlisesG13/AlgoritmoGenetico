"""
SAFE-CARGO | data_model.py
Clases base que representan todos los elementos del sistema.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# ENUMERACIONES
# ─────────────────────────────────────────────

class Categoria(Enum):
    ALIMENTO    = "alimento"
    LIQUIDO     = "liquido"
    FLORES      = "flores"
    DETERGENTE  = "detergente"
    HERRAMIENTA = "herramienta"


class TipoAmortiguador(Enum):
    NINGUNO          = "ninguno"
    CARTON           = "carton"
    PLASTICO_BURBUJA = "plastico_burbuja"
    ALGODON          = "algodon"


class CaraItem(Enum):
    TOP    = "top"
    BOTTOM = "bottom"
    LEFT   = "left"
    RIGHT  = "right"
    FRONT  = "front"
    BACK   = "back"


class ModoSujecion(Enum):
    NINGUNA      = "ninguna"
    LONGITUDINAL = "longitudinal"   # contra pendiente (eje Y)
    LATERAL      = "lateral"        # contra inclinacion (eje X)
    COMPLETA     = "completa"       # ambas direcciones


# ─────────────────────────────────────────────
# AMORTIGUADOR
# ─────────────────────────────────────────────

@dataclass
class Amortiguador:
    """
    Capa de material buffer que rodea un Item por las caras indicadas.
    No tiene posicion propia — se calcula a partir del Item que lo porta.
    """
    tipo            : TipoAmortiguador
    grosor          : float          # cm — ancho de la capa en cada cara activa
    coef_absorcion  : float          # 0.0 a 1.0
    caras_activas   : list[CaraItem] = field(default_factory=list)

    def dimensiones_extra(self) -> tuple[float, float, float]:
        """
        Retorna (dx, dy, dz) — cuanto se suma al item en cada eje
        dependiendo de las caras que cubre.
        """
        dx = self.grosor * (CaraItem.LEFT  in self.caras_activas) \
           + self.grosor * (CaraItem.RIGHT in self.caras_activas)
        dy = self.grosor * (CaraItem.FRONT in self.caras_activas) \
           + self.grosor * (CaraItem.BACK  in self.caras_activas)
        dz = self.grosor * (CaraItem.TOP    in self.caras_activas) \
           + self.grosor * (CaraItem.BOTTOM in self.caras_activas)
        return (dx, dy, dz)


# Instancias reutilizables de amortiguadores predefinidos
AMORTIGUADORES = {
    TipoAmortiguador.NINGUNO: Amortiguador(
        tipo           = TipoAmortiguador.NINGUNO,
        grosor         = 0.0,
        coef_absorcion = 0.0,
        caras_activas  = []
    ),
    TipoAmortiguador.CARTON: Amortiguador(
        tipo           = TipoAmortiguador.CARTON,
        grosor         = 1.0,   # cm
        coef_absorcion = 0.4,
        caras_activas  = []     # se asignan por item en el cromosoma
    ),
    TipoAmortiguador.PLASTICO_BURBUJA: Amortiguador(
        tipo           = TipoAmortiguador.PLASTICO_BURBUJA,
        grosor         = 2.0,
        coef_absorcion = 0.6,
        caras_activas  = []
    ),
    TipoAmortiguador.ALGODON: Amortiguador(
        tipo           = TipoAmortiguador.ALGODON,
        grosor         = 3.0,
        coef_absorcion = 0.8,
        caras_activas  = []
    ),
}


# ─────────────────────────────────────────────
# ITEM (mercancía)
# ─────────────────────────────────────────────

@dataclass
class Item:
    """
    Representa una unidad de mercancía a transportar.
    Dimensiones en cm, peso en kg.
    """
    id          : str
    nombre      : str
    largo       : float          # cm — eje Y
    ancho       : float          # cm — eje X
    alto        : float          # cm — eje Z
    peso        : float          # kg
    categoria   : Categoria
    fragilidad  : float          # 0.0 (nada frágil) a 1.0 (extremadamente frágil)
    apilable    : bool = True    # si puede soportar peso encima

    def volumen(self) -> float:
        return self.largo * self.ancho * self.alto

    def dimensiones(self) -> tuple[float, float, float]:
        """Retorna (largo, ancho, alto) — (Y, X, Z)."""
        return (self.largo, self.ancho, self.alto)


# ─────────────────────────────────────────────
# ZONA
# ─────────────────────────────────────────────

@dataclass
class Zona:
    """
    Region fisica delimitada dentro del vehiculo.
    Coordenadas en cm desde la esquina frontal-izquierda-inferior.
    """
    id              : str
    nombre          : str
    x_min           : float
    x_max           : float
    y_min           : float
    y_max           : float
    z_min           : float
    z_max           : float
    peso_max        : float               # kg maximos que soporta la zona
    categorias_prohibidas: list[Categoria] = field(default_factory=list)
    x_offset        : float = 0.0         # coordenada global X de la esquina local (0,0,0)
    y_offset        : float = 0.0         # coordenada global Y de la esquina local (0,0,0)
    z_offset        : float = 0.0         # coordenada global Z de la esquina local (0,0,0)

    def volumen(self) -> float:
        return (self.x_max - self.x_min) \
             * (self.y_max - self.y_min) \
             * (self.z_max - self.z_min)

    def contiene_punto(self, x: float, y: float, z: float) -> bool:
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def a_absoluto(self, x_local: float, y_local: float, z_local: float
                   ) -> tuple[float, float, float]:
        return (
            self.x_offset + x_local,
            self.y_offset + y_local,
            self.z_offset + z_local,
        )


# ─────────────────────────────────────────────
# VEHICULO
# ─────────────────────────────────────────────

@dataclass
class Vehiculo:
    """
    Plataforma de transporte compuesta por zonas.
    No guarda informacion del trayecto — eso vive en Trayecto.
    """
    id              : str
    nombre          : str
    zonas           : list[Zona]
    peso_max_total  : float     # kg — capacidad total del vehiculo

    def zona_por_id(self, zona_id: str) -> Optional[Zona]:
        for z in self.zonas:
            if z.id == zona_id:
                return z
        return None

    def volumen_total(self) -> float:
        return sum(z.volumen() for z in self.zonas)


# ─────────────────────────────────────────────
# TRAYECTO
# ─────────────────────────────────────────────

@dataclass
class Trayecto:
    """
    Condiciones topograficas de una ruta especifica.
    Un mismo Vehiculo puede usarse con distintos Trayectos.
    """
    id               : str
    nombre           : str
    inclinacion_long : float = 0.0   # grados — cabeceo (eje Y, pendiente adelante/atras)
    inclinacion_lat  : float = 0.0   # grados — alabeo  (eje X, inclinacion lateral)
    coef_vibracion   : float = 0.0   # 0.0 a 1.0 — irregularidad del terreno (0=asfalto, 1=terraceria brutal)


# ─────────────────────────────────────────────
# ORIENTACION
# ─────────────────────────────────────────────

@dataclass
class Orientacion:
    """
    Rotacion discreta de un item sobre la plataforma.
    theta_h: rotacion horizontal (giro sobre eje Z) — 0, 90, 180, 270
    theta_v: rotacion vertical   (giro sobre eje X) — 0, 90, 180, 270
    """
    theta_h: int = 0    # grados
    theta_v: int = 0    # grados

    VALORES_VALIDOS = (0, 90, 180, 270)

    def __post_init__(self):
        assert self.theta_h in self.VALORES_VALIDOS, \
            f"theta_h debe ser uno de {self.VALORES_VALIDOS}"
        assert self.theta_v in self.VALORES_VALIDOS, \
            f"theta_v debe ser uno de {self.VALORES_VALIDOS}"

    def aplicar(self, largo: float, ancho: float, alto: float
                ) -> tuple[float, float, float]:
        """
        Aplica la rotacion a las dimensiones del item y retorna
        las nuevas dimensiones (largo, ancho, alto) ya rotadas.
        """
        l, a, h = largo, ancho, alto

        # Rotacion vertical sobre eje X: intercambia alto y largo
        if self.theta_v == 90 or self.theta_v == 270:
            l, h = h, l

        # Rotacion horizontal sobre eje Z: intercambia largo y ancho
        if self.theta_h == 90 or self.theta_h == 270:
            l, a = a, l

        return (l, a, h)
