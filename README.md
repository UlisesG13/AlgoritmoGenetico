# SAFE-CARGO

> Optimización del empaquetado, distribución y sujeción de carga en vehículos de transporte mediante un Algoritmo Genético.

**SAFE-CARGO** es un sistema basado en cómputo evolutivo que resuelve el problema de cómo acomodar un conjunto de items dentro de un vehículo, decidiendo simultáneamente:

- En qué **zona** del vehículo va cada item.
- Su **posición** (x, y, z) en centímetros.
- Su **orientación** (rotación horizontal y vertical).
- El **tipo de amortiguador** que lo protegerá.
- El **modo de sujeción** que lo asegurará durante el trayecto.

La función de aptitud penaliza configuraciones inseguras (centro de masa descompensado, sujeciones incompatibles, items frágiles mal protegidos, sobrepeso por zona, etc.) y premia las soluciones estables, balanceadas y compatibles con el escenario topográfico elegido.

---

## Tabla de contenidos

- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Argumentos de línea de comandos](#argumentos-de-línea-de-comandos)
- [Parámetros del algoritmo genético](#parámetros-del-algoritmo-genético)
- [Salidas del sistema](#salidas-del-sistema)
- [Arquitectura interna](#arquitectura-interna)
- [Ejemplos de ejecución](#ejemplos-de-ejecución)
- [Autores](#autores)

---

## Características

- **Algoritmo Genético configurable** con población, generaciones, probabilidades de cruce/mutación y elitismo ajustables.
- **Dos operadores de cruce**: uniforme y un punto.
- **Dos métodos de selección**: torneo (por defecto) y ruleta.
- **Parada temprana por convergencia** con tolerancia y paciencia configurables.
- **Catálogo de items y trayectos predefinidos** en `knowledge_base.py`.
- **Reportes y visualizaciones** automatizadas: gráfica de convergencia, centro de masa, vista 3D de la carga, mapa de sujeción.
- **Exportación a PDF** del reporte completo de la mejor solución.
- **Reproducibilidad** garantizada mediante semilla aleatoria.

---

## Estructura del proyecto

```
AlgoritmoGenetico/
├── main.py              # Punto de entrada y CLI
├── genetic_engine.py    # Motor del AG (ciclo evolutivo, parámetros, resultados)
├── chromosome.py        # Representación del cromosoma y población inicial
├── operators.py         # Selección, cruce, mutación y elitismo
├── fitness.py           # Función de aptitud y pesos por criterio
├── data_model.py        # Entidades: Item, Vehiculo, Trayecto, Zona
├── knowledge_base.py    # Catálogo de items, trayectos y vehículo de prueba
└── visualization.py     # Gráficas, tablas comparativas y exportación a PDF
```

---

## Requisitos

- Python **3.10** o superior (usa `from __future__ import annotations` y sintaxis de tipos modernos).
- Bibliotecas estándar suficientes para el núcleo del AG.
- Para visualizaciones y reportes:
  - `matplotlib` — gráficas 2D y 3D.
  - `reportlab` *(o equivalente)* — exportación a PDF.

---

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/UlisesG13/AlgoritmoGenetico.git
cd AlgoritmoGenetico
```

Instala las dependencias (recomendado usar entorno virtual):

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install matplotlib reportlab
```

---

## Uso rápido

Ejecutar con configuración por defecto (6 items, trayecto moderado):

```bash
python main.py
```

Ejecutar con visualizaciones y exportación a PDF:

```bash
python main.py --visualizar --pdf
```

Ejecutar con items específicos del catálogo:

```bash
python main.py --items ITM01,ITM03,ITM07 --trayecto moderado
```

---

## Argumentos de línea de comandos

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--trayecto` | str | `moderado` | Escenario topográfico del viaje (definido en `TRAYECTOS`). |
| `--items` | str | — | IDs de items separados por coma. Si se omite, usa los primeros `--count` del catálogo. |
| `--count` | int | `6` | Número de items a usar si no se especifican IDs. |
| `--tam-poblacion` | int | `60` | Tamaño de la población. |
| `--max-generaciones` | int | `150` | Generaciones máximas a ejecutar. |
| `--semilla` | int | `42` | Semilla aleatoria para reproducibilidad. |
| `--tipo-cruce` | str | `uniforme` | Operador de cruce: `uniforme` o `un_punto`. |
| `--visualizar` | flag | `False` | Genera gráficas de convergencia, centro de masa, vista 3D y sujeción. |
| `--pdf` | flag | `False` | Exporta el reporte completo a `SAFE_CARGO_reporte.pdf`. |

---

## Parámetros del algoritmo genético

Los parámetros internos del AG están centralizados en la dataclass `ParametrosAG` (`genetic_engine.py`):

| Parámetro | Default | Descripción |
|---|---|---|
| `tam_poblacion` | 60 | Individuos por generación. |
| `max_generaciones` | 150 | Iteraciones máximas. |
| `prob_cruce` | 0.85 | Probabilidad de cruzar dos padres. |
| `tipo_cruce` | `"uniforme"` | `"uniforme"` o `"un_punto"`. |
| `tam_torneo` | 3 | Participantes en selección por torneo. |
| `usar_ruleta` | `False` | Si `True`, usa selección por ruleta. |
| `n_elite` | 2 | Individuos elite preservados cada generación. |
| `prob_mut_zona` | — | Probabilidad de mutar la zona de un gen. |
| `prob_mut_posicion` | — | Probabilidad de perturbar la posición. |
| `prob_mut_ori` | — | Probabilidad de mutar la orientación. |
| `prob_mut_amort` | — | Probabilidad de mutar el amortiguador. |
| `prob_mut_sujecion` | — | Probabilidad de mutar el modo de sujeción. |
| `tolerancia` | 1e-6 | Mejora mínima para no contar como estancamiento. |
| `paciencia` | 30 | Generaciones sin mejora antes de detener (0 = desactivado). |

---

## Salidas del sistema

Al finalizar la ejecución, el programa imprime y/o genera:

1. **Resumen de ejecución**: generaciones, tiempo total, convergencia anticipada, mejor aptitud.
2. **Top 3 soluciones** con detalle gen a gen (item, zona, posición, orientación, amortiguador, sujeción).
3. **Tabla comparativa** del top 3.
4. **Resumen de variables de decisión** de la mejor solución.
5. **Reporte de compatibilidad** items ↔ sujeciones ↔ amortiguadores.
6. *(Opcional con `--visualizar`)* Gráficas:
   - Convergencia de aptitud por generación.
   - Centro de masa de la mejor solución.
   - Vista 3D de la carga acomodada.
   - Mapa de sujeción por zona.
7. *(Opcional con `--pdf`)* PDF `SAFE_CARGO_reporte.pdf` con el reporte completo.

---

## Arquitectura interna

### Ciclo evolutivo

Para cada generación, `genetic_engine.ejecutar()` realiza:

1. Evaluar cromosomas sin aptitud asignada.
2. Registrar estadísticas (mejor, media, peor).
3. Verificar parada temprana por convergencia.
4. Seleccionar padres (torneo o ruleta).
5. Aplicar cruce (uniforme o un punto) según `prob_cruce`.
6. Mutar cada gen del hijo según probabilidades por atributo.
7. Evaluar la nueva población.
8. Aplicar **elitismo**: los `n_elite` mejores de la generación anterior pasan directamente.

### Representación del cromosoma

Cada **cromosoma** representa una solución completa de empaquetado. Cada **gen** describe la colocación de un item:

- `item_id` — identificador del item.
- `zona_id` — zona del vehículo asignada.
- `x, y, z` — posición en cm dentro de la zona.
- `orientacion` — ángulos horizontal y vertical.
- `tipo_amortiguador` — protección física aplicada.
- `modo_sujecion` — método de fijación.

### Función de aptitud

Definida en `fitness.py`, combina múltiples criterios ponderados:

- Balance del centro de masa respecto al vehículo.
- Compatibilidad item ↔ sujeción ↔ amortiguador.
- Adecuación al perfil del trayecto (curvas, pendientes, vibración).
- Penalizaciones por sobrepeso, colisiones o configuraciones inválidas.

Los pesos se configuran en `PesosAptitud` y pueden ajustarse antes de ejecutar.

---

## Ejemplos de ejecución

**Ejecución mínima**

```bash
python main.py
```

**Trayecto difícil con población grande y visualizaciones**

```bash
python main.py --trayecto difícil --tam-poblacion 120 --max-generaciones 300 --visualizar
```

**Comparar cruces sobre el mismo problema**

```bash
python main.py --tipo-cruce uniforme --semilla 42
python main.py --tipo-cruce un_punto --semilla 42
```

**Items específicos + reporte PDF**

```bash
python main.py --items ITM01,ITM02,ITM04,ITM08 --pdf
```

**Prueba rápida del motor sin pasar por la CLI**

```bash
python genetic_engine.py
```

Este modo de prueba ejecuta una configuración reducida (30 individuos, 50 generaciones) y verifica que el motor funcione correctamente.

---

## Autores

- **Ulises G.** — [@UlisesG13](https://github.com/UlisesG13)

---

## Licencia

Proyecto académico. Si vas a reutilizar el código, abre un issue o contacta a los autores.
