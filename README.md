# Raytracer con Golem 3D

Un programa de raytracing que renderiza un modelo 3D de Golem (Pokémon) construido completamente con esferas, implementando iluminación Phong realista con reflexiones y sombras.

## Características

### Renderizado
- **Raytracing en tiempo real** con renderizado inmediato
- **Iluminación Phong completa** (componentes ambiente, difusa y especular)
- **Reflexiones dinámicas** con profundidad configurable
- **Sombras realistas** calculadas por intersección de rayos
- **Múltiples fuentes de luz** para iluminación volumétrica

### Modelo 3D - Golem
- **Estructura anatómica** con proporciones correctas del Pokémon Golem
- **Perspectiva 3/4** para vista óptima del modelo
- **Geometría basada en esferas** (17 esferas en total):
  - Torso principal (esfera más grande)
  - Cabeza con ojos rojos brillantes
  - Extremidades articuladas (2 esferas por brazo/pierna)
  - Garras blancas detalladas (3 por pie)

### Materiales Intercambiables
- **Roca** (Material base): Textura mate con baja reflectividad
- **Metal**: Alta reflectividad y brillo especular intenso
- **Agua**: Reflectividad media con tonos azulados

## Controles

| Tecla | Función |
|-------|---------|
| `1` | Cambiar a material de Roca |
| `2` | Cambiar a material de Metal |
| `3` | Cambiar a material de Agua |
| `ENTER` | Guardar imagen como archivo BMP |
| `ESC` | Salir del programa |

## Tecnologías

- **Python 3** con Pygame para la interfaz gráfica
- **Raytracing puro** implementado desde cero
- **Matemáticas vectoriales** para cálculos de intersección
- **Algoritmo de Phong** para iluminación realista

## Ejecución

```bash
python main_golem.py