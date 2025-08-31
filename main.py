import pygame
import os
import math
from gl import *



width = 512
height = 512
pygame.init()
screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()
rend = Renderer(screen)

# Configurar la escena de Golem
rend.create_sphere_scene()

# Activar modo raytracing automáticamente
rend.primitiveType = RAYTRACER

# Configurar cámara
rend.set_camera((0, 0, 6), (0, 0, 0), (0, 1, 0))

print("RayTracer - Golem de Pokémon")
print("Controles:")
print("- ESC: Salir")
print("- ENTER: Guardar imagen")
print("- 1: Material de Roca")
print("- 2: Material de Metal") 
print("- 3: Material de Agua")

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_RETURN:
                material_names = ["roca", "metal", "agua"]
                filename = f"golem_{material_names[rend.current_material_type]}.bmp"
                rend.save_bmp(filename)
                print(f"Imagen guardada como {filename}")
            elif e.key == pygame.K_1:
                rend.change_material(0)  # Roca
            elif e.key == pygame.K_2:
                rend.change_material(1)  # Metal
            elif e.key == pygame.K_3:
                rend.change_material(2)  # Agua

    rend.glClear()
    rend.glRender()

    material_names = ["Roca", "Metal", "Agua"]
    pygame.display.set_caption(f"Golem - Material: {material_names[rend.current_material_type]} | ESC: Salir | 1-3: Cambiar Material")
    pygame.display.flip()
    clock.tick(60)

pygame.quit()