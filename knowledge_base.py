

from data_model import (
    Item, Zona, Vehiculo, Trayecto,
    Amortiguador, AMORTIGUADORES,
    Categoria, TipoAmortiguador, CaraItem
)



MATRIZ_COMPATIBILIDAD: dict[tuple[Categoria, Categoria], bool] = {
    (Categoria.ALIMENTO,    Categoria.ALIMENTO)    : True,
    (Categoria.ALIMENTO,    Categoria.LIQUIDO)     : True,
    (Categoria.ALIMENTO,    Categoria.FLORES)      : True,
    (Categoria.ALIMENTO,    Categoria.DETERGENTE)  : False,
    (Categoria.ALIMENTO,    Categoria.HERRAMIENTA) : False,
    (Categoria.LIQUIDO,     Categoria.ALIMENTO)    : True,
    (Categoria.LIQUIDO,     Categoria.LIQUIDO)     : True,
    (Categoria.LIQUIDO,     Categoria.FLORES)      : True,
    (Categoria.LIQUIDO,     Categoria.DETERGENTE)  : True,
    (Categoria.LIQUIDO,     Categoria.HERRAMIENTA) : False,
    (Categoria.FLORES,      Categoria.ALIMENTO)    : True,
    (Categoria.FLORES,      Categoria.LIQUIDO)     : True,
    (Categoria.FLORES,      Categoria.FLORES)      : True,
    (Categoria.FLORES,      Categoria.DETERGENTE)  : False,
    (Categoria.FLORES,      Categoria.HERRAMIENTA) : False,
    (Categoria.DETERGENTE,  Categoria.ALIMENTO)    : False,
    (Categoria.DETERGENTE,  Categoria.LIQUIDO)     : True,
    (Categoria.DETERGENTE,  Categoria.FLORES)      : False,
    (Categoria.DETERGENTE,  Categoria.DETERGENTE)  : True,
    (Categoria.DETERGENTE,  Categoria.HERRAMIENTA) : False,
    (Categoria.HERRAMIENTA, Categoria.ALIMENTO)    : False,
    (Categoria.HERRAMIENTA, Categoria.LIQUIDO)     : False,
    (Categoria.HERRAMIENTA, Categoria.FLORES)      : False,
    (Categoria.HERRAMIENTA, Categoria.DETERGENTE)  : False,
    (Categoria.HERRAMIENTA, Categoria.HERRAMIENTA) : True,
}


def son_compatibles(cat_a: Categoria, cat_b: Categoria) -> bool:
    """Consulta si dos categorias pueden estar en proximidad."""
    return MATRIZ_COMPATIBILIDAD.get((cat_a, cat_b), False)


# ─────────────────────────────────────────────
# CATALOGO DE ITEMS DE PRUEBA
# largo x ancho x alto en cm | peso en kg
# fragilidad: 0.0 = nada fragil, 1.0 = extremadamente fragil
# ─────────────────────────────────────────────

CATALOGO_ITEMS: list[Item] = [
    Item(id="A01", nombre="Caja de jitomates",
         largo=40, ancho=30, alto=20, peso=15.0,
         categoria=Categoria.ALIMENTO, fragilidad=0.7, apilable=False),
    Item(id="A02", nombre="Costal de maiz",
         largo=80, ancho=50, alto=20, peso=50.0,
         categoria=Categoria.ALIMENTO, fragilidad=0.1, apilable=True),
    Item(id="A03", nombre="Caja de aguacates",
         largo=50, ancho=30, alto=25, peso=18.0,
         categoria=Categoria.ALIMENTO, fragilidad=0.5, apilable=False),
    Item(id="A04", nombre="Canasta de huevos",
         largo=35, ancho=35, alto=20, peso=8.0,
         categoria=Categoria.ALIMENTO, fragilidad=0.95, apilable=False),
    Item(id="L01", nombre="Garrafon de agua (20L)",
         largo=30, ancho=30, alto=45, peso=20.0,
         categoria=Categoria.LIQUIDO, fragilidad=0.3, apilable=False),
    Item(id="L02", nombre="Cubeta de miel (19L)",
         largo=30, ancho=30, alto=35, peso=27.0,
         categoria=Categoria.LIQUIDO, fragilidad=0.4, apilable=True),
    Item(id="L03", nombre="Botella de aceite de oliva",
         largo=10, ancho=10, alto=30, peso=1.5,
         categoria=Categoria.LIQUIDO, fragilidad=0.8, apilable=False),
    Item(id="F01", nombre="Ramo de rosas (docena)",
         largo=20, ancho=15, alto=60, peso=1.0,
         categoria=Categoria.FLORES, fragilidad=0.9, apilable=False),
    Item(id="F02", nombre="Maceta de geranios",
         largo=25, ancho=25, alto=30, peso=3.5,
         categoria=Categoria.FLORES, fragilidad=0.75, apilable=False),
    Item(id="D01", nombre="Caja de jabon en polvo (5kg)",
         largo=35, ancho=20, alto=25, peso=5.5,
         categoria=Categoria.DETERGENTE, fragilidad=0.1, apilable=True),
    Item(id="D02", nombre="Garrafon de cloro (20L)",
         largo=30, ancho=30, alto=45, peso=21.0,
         categoria=Categoria.DETERGENTE, fragilidad=0.3, apilable=False),
    Item(id="H01", nombre="Hacha de tala",
         largo=70, ancho=10, alto=10, peso=3.5,
         categoria=Categoria.HERRAMIENTA, fragilidad=0.0, apilable=True),
    Item(id="H02", nombre="Tijeras de podar",
         largo=25, ancho=8,  alto=5,  peso=0.8,
         categoria=Categoria.HERRAMIENTA, fragilidad=0.0, apilable=True),
    Item(id="H03", nombre="Pico de campo",
         largo=100, ancho=10, alto=8, peso=4.0,
         categoria=Categoria.HERRAMIENTA, fragilidad=0.0, apilable=True),
    Item(id="H04", nombre="Manguera enrollada",
         largo=40, ancho=40, alto=15, peso=5.0,
         categoria=Categoria.HERRAMIENTA, fragilidad=0.1, apilable=True),
]


def item_por_id(item_id: str) -> Item | None:
    """Busca un item en el catalogo por su id."""
    return next((i for i in CATALOGO_ITEMS if i.id == item_id), None)


# ─────────────────────────────────────────────
# VEHICULO Y ZONAS DE PRUEBA
# Pickup Ford Ranger doble cabina
# Todas las medidas en cm
#
# Vista superior:
# ┌─────────────────┬──────────────────────┐
# │    Z_CABINA     │      Z_BATEA         │
# │  (asiento tras) │  (plataforma carga)  │
# │   130 x 80      │    130 x 150         │
# └─────────────────┴──────────────────────┘
# Z_TECHO (rack sobre cabina): 130 x 80
#
# IMPORTANTE: coordenadas de cada item son LOCALES a su zona
#   (0,0,0) = esquina frontal-izquierda-inferior de cada zona
#   Eje X = ancho (izquierda a derecha)
#   Eje Y = largo (frente hacia atras)
#   Eje Z = altura (piso hacia arriba)
# ─────────────────────────────────────────────

def crear_vehiculo_prueba() -> Vehiculo:
    zonas = [
        Zona(
            id="Z_CABINA", nombre="Asiento trasero cabina",
            x_min=0,   x_max=130,
            y_min=0,   y_max=80,
            z_min=0,   z_max=90,
            peso_max=80,
            categorias_prohibidas=[Categoria.DETERGENTE, Categoria.HERRAMIENTA],
            x_offset=0.0,
            y_offset=0.0,
            z_offset=0.0,
        ),
        Zona(
            id="Z_BATEA", nombre="Batea trasera",
            x_min=0,   x_max=130,
            y_min=0,   y_max=150,
            z_min=0,   z_max=50,
            peso_max=800,
            categorias_prohibidas=[],
            x_offset=0.0,
            y_offset=80.0,
            z_offset=0.0,
        ),
        Zona(
            id="Z_TECHO", nombre="Rack techo cabina",
            x_min=0,   x_max=130,
            y_min=0,   y_max=80,
            z_min=0,   z_max=50,
            peso_max=100,
            categorias_prohibidas=[
                Categoria.LIQUIDO,
                Categoria.FLORES,
                Categoria.ALIMENTO
            ],
            x_offset=0.0,
            y_offset=0.0,
            z_offset=90.0,
        ),
    ]
    return Vehiculo(
        id="V01",
        nombre="Pickup Ford Ranger doble cabina",
        zonas=zonas,
        peso_max_total=900,
    )


# TRAYECTOS DE PRUEBA

TRAYECTOS: dict[str, Trayecto] = {
    "plano": Trayecto(
        id="T01", nombre="Carretera pavimentada",
        inclinacion_long=2.0, inclinacion_lat=1.0, coef_vibracion=0.1
    ),
    "moderado": Trayecto(
        id="T02", nombre="Camino rural compactado",
        inclinacion_long=10.0, inclinacion_lat=5.0, coef_vibracion=0.5
    ),
    "severo": Trayecto(
        id="T03", nombre="Terraceria de montana",
        inclinacion_long=20.0, inclinacion_lat=12.0, coef_vibracion=0.9
    ),
}
