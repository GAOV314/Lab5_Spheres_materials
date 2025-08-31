import pygame
import math
import random

POINTS = 0
LINES = 1
TRIANGLES = 2
RAYTRACER = 3 

# ============= CLASES PARA RAYTRACING =============

class Material:
    def __init__(self, diffuse=(1,1,1), specular=(1,1,1), ambient=(0.1,0.1,0.1), shininess=32, reflectivity=0.0):
        self.diffuse = diffuse     
        self.specular = specular    
        self.ambient = ambient      
        self.shininess = shininess  
        self.reflectivity = reflectivity  

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center     
        self.radius = radius     
        self.material = material  
    
    def ray_intersect(self, ray_origin, ray_direction):
    
        # Vector del origen del rayo al centro de la esfera
        oc = [ray_origin[i] - self.center[i] for i in range(3)]
        
        # Coeficientes de la ecuación cuadrática
        a = sum(ray_direction[i] * ray_direction[i] for i in range(3))
        b = 2.0 * sum(oc[i] * ray_direction[i] for i in range(3))
        c = sum(oc[i] * oc[i] for i in range(3)) - self.radius * self.radius
        
        # Discriminante
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None  
        
        sqrt_discriminant = math.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        
        if t1 > 0.001:  
            return t1
        elif t2 > 0.001:
            return t2
        else:
            return None
    
    def get_normal(self, point):
        """Obtiene la normal en un punto de la superficie de la esfera"""
        normal = [(point[i] - self.center[i]) / self.radius for i in range(3)]
        return normal

class Light:
    def __init__(self, position, color=(1,1,1), intensity=1.0):
        self.position = position
        self.color = color
        self.intensity = intensity

# ============= RENDERER PRINCIPAL =============

class Renderer(object):
    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = self.screen.get_rect()
        
        # Inicializar variables básicas PRIMERO
        self.primitiveType = TRIANGLES
        self.models = []
        
        # Matrices / cámara
        self.view_matrix = self.identity()
        self.proj_matrix = self.identity()
        self.fov_deg = 60
        self.near = 0.1
        self.far = 1000.0

        # Shader mode
        self.shader_mode = 0
        self.frame_count = 0

        # ============= RAYTRACING =============
        self.spheres = []
        self.lights = []
        self.camera_pos = (0, 0, 0)
        self.background_color = (0.1, 0.1, 0.2)
        self.max_depth = 3  # Profundidad máxima de reflexión
        
        # Control de renderizado de raytracing
        self.raytracing_completed = False
        
        # Control de materiales
        self.current_material_type = 0  # 0=roca, 1=metal, 2=agua
        
        # Ahora sí inicializar colores y limpiar
        self.glColor(1,1,1)
        self.glClearColor(0.1,0.1,0.15)
        self.glClear()

    # ============= FUNCIONES BÁSICAS =============
    
    def identity(self):
        return [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]

    def multiply(self, A,B):
        r = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r[i][j] += A[i][k]*B[k][j]
        return r

    def vec4_mul_mat(self, M, v):
        return [
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3],
            M[3][0]*v[0] + M[3][1]*v[1] + M[3][2]*v[2] + M[3][3]*v[3],
        ]

    def normalize_vec3(self, v):
        l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if l == 0:
            return [0,0,0]
        return [v[0]/l, v[1]/l, v[2]/l]

    def cross(self, a,b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]

    def look_at(self, eye, target, up):
        zaxis = self.normalize_vec3([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]])
        xaxis = self.normalize_vec3(self.cross(up, zaxis))
        yaxis = self.cross(zaxis, xaxis)
        return [
            [xaxis[0], xaxis[1], xaxis[2], - (xaxis[0]*eye[0] + xaxis[1]*eye[1] + xaxis[2]*eye[2])],
            [yaxis[0], yaxis[1], yaxis[2], - (yaxis[0]*eye[0] + yaxis[1]*eye[1] + yaxis[2]*eye[2])],
            [zaxis[0], zaxis[1], zaxis[2], - (zaxis[0]*eye[0] + zaxis[1]*eye[1] + zaxis[2]*eye[2])],
            [0,0,0,1]
        ]

    def perspective(self, fov_deg, aspect, near, far):
        f = 1 / math.tan(math.radians(fov_deg)/2)
        nf = 1 / (near - far)
        return [
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)*nf, (2*far*near)*nf],
            [0, 0, -1, 0]
        ]

    # ============= FUNCIONES GL =============
    
    def glClearColor(self, r,g,b):
        self.clearColor = [max(0,min(1,r)),max(0,min(1,g)),max(0,min(1,b))]

    def glColor(self, r,g,b):
        self.currColor = [max(0,min(1,r)),max(0,min(1,g)),max(0,min(1,b))]

    def glClear(self):
        # Solo limpiar si no estamos en modo raytracing o si no está completo
        if self.primitiveType != RAYTRACER or not self.raytracing_completed:
            c = [int(i*255) for i in self.clearColor]
            self.screen.fill(c)
            self.frameBuffer = [[self.clearColor for _ in range(self.height)] for _ in range(self.width)]
            
            # Si cambiamos a raytracing, marcar como no completado
            if self.primitiveType == RAYTRACER:
                self.raytracing_completed = False

    def glPoint(self, x,y, color=None):
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < self.width and 0 <= y < self.height:
            src = color or self.currColor
            col = [int(max(0,min(1,c))*255) for c in src]
            self.screen.set_at((x, self.height - 1 - y), col)
            self.frameBuffer[x][y] = [c/255 for c in col]

    def glLine(self, p0, p1, color=None):
        x0,y0 = p0; x1,y1 = p1
        dx = abs(x1-x0); dy = abs(y1-y0)
        steep = dy > dx
        if steep:
            x0,y0 = y0,x0
            x1,y1 = y1,x1
        if x0 > x1:
            x0,x1 = x1,x0
            y0,y1 = y1,y0
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = 0
        y = y0
        ystep = 1 if y0 < y1 else -1
        for x in range(x0, x1+1):
            if steep:
                self.glPoint(y,x,color)
            else:
                self.glPoint(x,y,color)
            error += dy
            if (error * 2) >= dx:
                y += ystep
                error -= dx

    # ============= FUNCIONES DE PROYECCIÓN =============
    
    def project_vertex(self, v4):
        v_view = self.vec4_mul_mat(self.view_matrix, v4)
        clip = self.vec4_mul_mat(self.proj_matrix, v_view)
        w = clip[3]
        if abs(w) < 1e-8:
            return None
        ndc_x = clip[0] / w
        ndc_y = clip[1] / w
        ndc_z = clip[2] / w
        sx = (ndc_x + 1) * 0.5 * self.width
        sy = (ndc_y + 1) * 0.5 * self.height
        return (sx, sy, ndc_z)

    # ============= SHADERS EXISTENTES =============
    
    def shader_hologram(self, base_color, u,v, x,y, z, time):
        scan = 0.5 + 0.5 * math.sin((y * 0.15) + time * 6.0)
        flicker = 0.85 + 0.15 * math.sin(time * 20 + x * 0.05)
        r = base_color[0] * 0.4
        g = min(1.0, base_color[1] * 0.9 + 0.3)
        b = min(1.0, base_color[2] * 1.0 + 0.4)
        factor = scan * flicker
        return [min(1,r*factor), min(1,g*factor), min(1,b*factor)]

    def shader_xray(self, base_color, u,v, x,y, z, bary):
        edge = min(bary[0], bary[1], bary[2])
        edge_threshold = 0.06
        depth_intensity = 1.0 - max(0.0, min(1.0, (z + 1) / 2))
        if edge < edge_threshold:
            return [0.2 + 0.8*depth_intensity, 0.8*depth_intensity, 1.0]
        gray = (base_color[0] + base_color[1] + base_color[2]) / 3
        return [gray*0.4, gray*0.8, gray*depth_intensity]

    def shader_water(self, model, base_color, u,v, x,y, z, time):
        if model.texture is None:
            base_color = [0.2, 0.4, 0.7]

        wave_speed1 = 0.9
        wave_speed2 = 1.3
        freq1 = 6.0
        freq2 = 9.5
        amp_uv = 0.015  
        
        du = amp_uv * math.sin(freq1 * (u + time * wave_speed1)) \
           + amp_uv * 0.6 * math.sin(freq2 * (v + time * wave_speed2) + 1.2)
        dv = amp_uv * math.sin(freq1 * (v + time * wave_speed1) + 0.7) \
           + amp_uv * 0.6 * math.sin(freq2 * (u + time * wave_speed2) + 2.3)

        u2 = u + du
        v2 = v + dv

        tex_color = model.get_texture_color(u2, v2) if model.texture else base_color[:]

        amp_h = 0.05
        h = (
            amp_h * math.sin(freq1 * (u + time * wave_speed1)) +
            amp_h * 0.6 * math.sin(freq2 * (v + time * wave_speed2) + 1.2)
        )

        eps = 0.002
        h_du = (
            amp_h * math.cos(freq1 * (u + eps + time * wave_speed1)) * freq1 -
            amp_h * math.cos(freq1 * (u + time * wave_speed1)) * freq1
        ) / eps
        h_dv = (
            amp_h * 0.6 * math.cos(freq2 * (v + eps + time * wave_speed2) + 1.2) * freq2 -
            amp_h * 0.6 * math.cos(freq2 * (v + time * wave_speed2) + 1.2) * freq2
        ) / eps

        nx = -h_du
        ny = -h_dv
        nz = 1.0
        nl = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nl > 0:
            nx /= nl; ny /= nl; nz /= nl

        L = self.normalize_vec3([0.3, 0.7, 0.6])
        NdotL = max(0.0, nx*L[0] + ny*L[1] + nz*L[2])

        diffuse = 0.4 + 0.6 * NdotL  

        V = [0,0,1]
        H = self.normalize_vec3([L[0]+V[0], L[1]+V[1], L[2]+V[2]])
        NdotH = max(0.0, nx*H[0] + ny*H[1] + nz*H[2])
        spec = pow(NdotH, 32) * 0.6  

        water_tint = [0.2, 0.45, 0.75]

        mixed = [
            tex_color[0] * water_tint[0],
            tex_color[1] * water_tint[1],
            tex_color[2] * water_tint[2]
        ]

        shaded = [
            mixed[0] * diffuse + spec,
            mixed[1] * diffuse + spec * 0.9,
            mixed[2] * (diffuse + spec * 0.8)
        ]

        for i in range(3):
            if shaded[i] > 1:
                shaded[i] = 1 - (shaded[i]-1)*0.5

        return [max(0,min(1,shaded[0])),
                max(0,min(1,shaded[1])),
                max(0,min(1,shaded[2]))]

    def shader_noise(self, base_color, u,v, x,y, z, frame):
        seed = int(x)*374761393 + int(y)*668265263 + frame*69069
        seed = (seed ^ (seed >> 13)) & 0xFFFFFFFF
        n = ((seed * 1274126177) & 0xFFFFFFFF) / 0xFFFFFFFF
        noise_strength = 0.45
        return [
            base_color[0]*(1-noise_strength) + n*noise_strength,
            base_color[1]*(1-noise_strength) + n*noise_strength,
            base_color[2]*(1-noise_strength) + n*noise_strength
        ]

    def apply_shader(self, mode, model, base_color, u,v, x,y, z, bary=None):
        if mode == 0:
            col = self.shader_hologram(base_color, u,v, x,y, z, self.frame_count/60.0)
        elif mode == 1:
            col = self.shader_xray(base_color, u,v, x,y, z, bary)
        elif mode == 2:
            col = self.shader_water(model, base_color, u,v, x,y, z, self.frame_count/60.0)
        elif mode == 3:
            col = self.shader_noise(base_color, u,v, x,y, z, self.frame_count)
        else:
            col = base_color
        return [max(0.0, min(1.0, col[0])),
                max(0.0, min(1.0, col[1])),
                max(0.0, min(1.0, col[2]))]

    # ============= FUNCIONES DE RASTERIZACIÓN =============
    
    def barycentric(self, x,y, A,B,C):
        denom = (B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1])
        if abs(denom) < 1e-10:
            return None
        a = ((B[1]-C[1])*(x-C[0]) + (C[0]-B[0])*(y-C[1])) / denom
        b = ((C[1]-A[1])*(x-C[0]) + (A[0]-C[0])*(y-C[1])) / denom
        c = 1 - a - b
        return a,b,c

    def draw_textured_triangle(self, A,B,C, uvA,uvB,uvC, zA,zB,zC, model):
        min_x = max(0, int(min(A[0],B[0],C[0])))
        max_x = min(self.width-1, int(max(A[0],B[0],C[0])))
        min_y = max(0, int(min(A[1],B[1],C[1])))
        max_y = min(self.height-1, int(max(A[1],B[1],C[1])))

        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                bc = self.barycentric(x,y, A,B,C)
                if bc is None:
                    continue
                a,b,c = bc
                if a >= 0 and b >= 0 and c >= 0:
                    u = a*uvA[0] + b*uvB[0] + c*uvC[0]
                    v = a*uvA[1] + b*uvB[1] + c*uvC[1]
                    z = a*zA + b*zB + c*zC
                    base_color = model.get_texture_color(u,v) if model.texture else [
                        random.uniform(0.3,1.0),
                        random.uniform(0.3,1.0),
                        random.uniform(0.3,1.0)
                    ]
                    shaded = self.apply_shader(self.shader_mode, model, base_color, u,v, x,y, z, (a,b,c))
                    self.glPoint(x,y, shaded)

    # ============= FUNCIONES DE RAYTRACING =============
    
    def add_sphere(self, sphere):
        """Añade una esfera a la escena de raytracing"""
        self.spheres.append(sphere)
    
    def add_light(self, light):
        """Añade una luz a la escena de raytracing"""
        self.lights.append(light)
    
    def set_raytracing_camera(self, position):
        """Establece la posición de la cámara para raytracing"""
        self.camera_pos = position
    
    def dot_product(self, a, b):
        """Producto punto entre dos vectores"""
        return sum(a[i] * b[i] for i in range(3))
    
    def vector_subtract(self, a, b):
        """Resta de vectores a - b"""
        return [a[i] - b[i] for i in range(3)]
    
    def vector_add(self, a, b):
        """Suma de vectores a + b"""
        return [a[i] + b[i] for i in range(3)]
    
    def vector_scale(self, vector, scalar):
        """Multiplica un vector por un escalar"""
        return [vector[i] * scalar for i in range(3)]
    
    def reflect(self, incident, normal):
        """Calcula el vector de reflexión"""
        # R = I - 2 * (I · N) * N
        dot = self.dot_product(incident, normal)
        return self.vector_subtract(incident, self.vector_scale(normal, 2 * dot))
    
    def ray_intersect_scene(self, ray_origin, ray_direction):
        """Encuentra la intersección más cercana con todas las esferas"""
        closest_t = float('inf')
        closest_sphere = None
        
        for sphere in self.spheres:
            t = sphere.ray_intersect(ray_origin, ray_direction)
            if t is not None and t < closest_t:
                closest_t = t
                closest_sphere = sphere
        
        if closest_sphere is None:
            return None, None
        
        return closest_t, closest_sphere
    
    def is_in_shadow(self, point, light_pos):
        """Verifica si un punto está en sombra respecto a una luz"""
        light_dir = self.normalize_vec3(self.vector_subtract(light_pos, point))
        shadow_ray_origin = self.vector_add(point, self.vector_scale(light_dir, 0.001))
        
        t, _ = self.ray_intersect_scene(shadow_ray_origin, light_dir)
        
        if t is not None:
            # Calcular la distancia a la luz
            light_distance = math.sqrt(sum((light_pos[i] - point[i])**2 for i in range(3)))
            return t < light_distance
        
        return False
    
    def phong_shading(self, point, normal, view_dir, material):
        """Implementa el modelo de iluminación Phong"""
        color = [0, 0, 0]
        
        # Componente ambiente
        ambient = [material.ambient[i] * 0.3 for i in range(3)]
        color = self.vector_add(color, ambient)
        
        for light in self.lights:
            # Verificar si está en sombra
            if self.is_in_shadow(point, light.position):
                continue
            
            # Vector hacia la luz
            light_dir = self.normalize_vec3(self.vector_subtract(light.position, point))
            
            # Componente difusa (Lambert)
            n_dot_l = max(0, self.dot_product(normal, light_dir))
            diffuse = [
                material.diffuse[i] * light.color[i] * light.intensity * n_dot_l
                for i in range(3)
            ]
            color = self.vector_add(color, diffuse)
            
            # Componente especular (Phong)
            if n_dot_l > 0:
                reflect_dir = self.reflect(self.vector_scale(light_dir, -1), normal)
                r_dot_v = max(0, self.dot_product(reflect_dir, view_dir))
                specular_intensity = pow(r_dot_v, material.shininess)
                specular = [
                    material.specular[i] * light.color[i] * light.intensity * specular_intensity
                    for i in range(3)
                ]
                color = self.vector_add(color, specular)
        
        # Clamp color values
        return [min(1, max(0, c)) for c in color]
    
    def trace_ray(self, ray_origin, ray_direction, depth=0):
        """Traza un rayo y calcula el color"""
        if depth >= self.max_depth:
            return self.background_color
        
        t, sphere = self.ray_intersect_scene(ray_origin, ray_direction)
        
        if sphere is None:
            return self.background_color
        
        # Punto de intersección
        hit_point = [ray_origin[i] + t * ray_direction[i] for i in range(3)]
        
        # Normal en el punto de intersección
        normal = sphere.get_normal(hit_point)
        
        # Vector hacia la cámara
        view_dir = self.normalize_vec3(self.vector_subtract(ray_origin, hit_point))
        
        # Calcular color usando Phong
        color = self.phong_shading(hit_point, normal, view_dir, sphere.material)
        
        # Reflexión
        if sphere.material.reflectivity > 0:
            reflect_dir = self.reflect(self.vector_scale(ray_direction, -1), normal)
            reflect_origin = self.vector_add(hit_point, self.vector_scale(normal, 0.001))
            reflect_color = self.trace_ray(reflect_origin, reflect_dir, depth + 1)
            
            # Mezclar color directo con reflexión
            for i in range(3):
                color[i] = color[i] * (1 - sphere.material.reflectivity) + \
                          reflect_color[i] * sphere.material.reflectivity
        
        return color
    
    def render_raytracing_immediate(self):
        """Renderiza raytracing de forma inmediata completa"""
        if self.raytracing_completed:
            return
        
        print("Renderizando raytracing...")
        
        fov = self.fov_deg
        aspect_ratio = self.width / self.height
        scale = math.tan(math.radians(fov * 0.5))
        
        # Renderizar toda la imagen de una vez
        for y in range(self.height):
            for x in range(self.width):
                # Convertir coordenadas de pantalla a coordenadas normalizadas
                px = (2 * (x + 0.5) / self.width - 1) * aspect_ratio * scale
                py = (1 - 2 * (y + 0.5) / self.height) * scale
                
                # Dirección del rayo
                ray_direction = self.normalize_vec3([px, py, -1])
                
                # Trazar el rayo
                color = self.trace_ray(self.camera_pos, ray_direction)
                
                # Dibujar el pixel
                self.glPoint(x, y, color)
        
        self.raytracing_completed = True
        print("¡Raytracing completado!")

    # ============= FUNCIÓN PRINCIPAL DE RENDERIZADO =============

    def glRender(self):
        # Si está en modo raytracing, renderizar las esferas inmediatamente
        if self.primitiveType == RAYTRACER:
            self.render_raytracing_immediate()
            self.frame_count += 1
            return
        
        # Renderizado normal de modelos
        for model in self.models:
            tris = model.get_triangles()

            if self.primitiveType == POINTS:
                for tri in tris:
                    for (x,y,z,u,v) in tri:
                        pv = self.project_vertex((x,y,z,1))
                        if pv:
                            self.glPoint(pv[0], pv[1], [1,1,1])
                self.frame_count += 1
                continue

            if self.primitiveType == LINES:
                for tri in tris:
                    pts2d = []
                    for (x,y,z,u,v) in tri:
                        pv = self.project_vertex((x,y,z,1))
                        if pv:
                            pts2d.append(pv)
                    if len(pts2d) == 3:
                        self.glLine((pts2d[0][0], pts2d[0][1]), (pts2d[1][0], pts2d[1][1]), [1,1,1])
                        self.glLine((pts2d[1][0], pts2d[1][1]), (pts2d[2][0], pts2d[2][1]), [1,1,1])
                        self.glLine((pts2d[2][0], pts2d[2][1]), (pts2d[0][0], pts2d[0][1]), [1,1,1])
                self.frame_count += 1
                continue

            # Renderizado de triángulos
            for tri in tris:
                proj_pts = []
                uvs = []
                for (x,y,z,u,v) in tri:
                    pv = self.project_vertex((x,y,z,1))
                    if not pv:
                        break
                    proj_pts.append(pv)
                    uvs.append((u,v))
                if len(proj_pts) == 3:
                    A = (proj_pts[0][0], proj_pts[0][1])
                    B = (proj_pts[1][0], proj_pts[1][1])
                    C = (proj_pts[2][0], proj_pts[2][1])
                    self.draw_textured_triangle(
                        A,B,C,
                        uvs[0],uvs[1],uvs[2],
                        proj_pts[0][2], proj_pts[1][2], proj_pts[2][2],
                        model
                    )

        self.frame_count += 1

    # ============= FUNCIONES DE UTILIDAD =============
    
    def set_camera(self, eye, target, up):
        aspect = self.width / self.height
        self.view_matrix = self.look_at(eye, target, up)
        self.proj_matrix = self.perspective(self.fov_deg, aspect, self.near, self.far)
        # También actualizar la posición de la cámara para raytracing
        self.camera_pos = eye
        
        # Si estamos en modo raytracing, marcar para re-renderizar
        if self.primitiveType == RAYTRACER:
            self.raytracing_completed = False

    def get_material_by_type(self, material_type):
        """Retorna el material según el tipo seleccionado"""
        if material_type == 0:  # Roca
            return Material(
                diffuse=(0.5, 0.4, 0.3),
                specular=(0.3, 0.3, 0.3),
                ambient=(0.05, 0.04, 0.03),
                shininess=8,
                reflectivity=0.05
            )
        elif material_type == 1:  # Metal
            return Material(
                diffuse=(0.6, 0.6, 0.7),
                specular=(1, 1, 1),
                ambient=(0.06, 0.06, 0.07),
                shininess=128,
                reflectivity=0.6
            )
        elif material_type == 2:  # Agua
            return Material(
                diffuse=(0.2, 0.4, 0.8),
                specular=(1, 1, 1),
                ambient=(0.02, 0.04, 0.08),
                shininess=64,
                reflectivity=0.4
            )
        else:
            return Material()  # Material por defecto

    def change_material(self, material_type):
        """Cambia el material de todas las esferas (excepto ojos y garras)"""
        self.current_material_type = material_type
        new_material = self.get_material_by_type(material_type)
        
        # Cambiar material solo de las esferas del cuerpo (no ojos ni garras)
        for i, sphere in enumerate(self.spheres):
            # Los primeros elementos son el cuerpo, los ojos están al final
            if i < len(self.spheres) - 8:  # Preservar 8 esferas (2 ojos + 6 garras)
                sphere.material = new_material
        
        # Marcar para re-renderizar
        if self.primitiveType == RAYTRACER:
            self.raytracing_completed = False
        
        material_names = ["Roca", "Metal", "Agua"]
        print(f"Material cambiado a: {material_names[material_type]}")

    def create_sphere_scene(self):
        # Limpiar escena
        self.spheres = []
        self.lights = []
        
        # Material principal del cuerpo
        body_material = self.get_material_by_type(self.current_material_type)
        
        # ============= TORSO/CAPARAZÓN (ESFERA GRANDE) =============
        # Esta es la esfera más grande del cuerpo - CENTRO
        self.add_sphere(Sphere((0, -0.1, -3), 0.9, body_material))
        
        # ============= CABEZA =============
        # Cabeza más pequeña que el torso, posicionada al frente y ARRIBA (Y negativa = arriba)
        self.add_sphere(Sphere((0.2, -0.9, -2.3), 0.5, body_material))
        
        # ============= BRAZO IZQUIERDO (perspectiva 3/4) =============
        # Hombro/brazo superior - a la altura del torso
        self.add_sphere(Sphere((-0.8, -0.3, -2.8), 0.35, body_material))
        # Antebrazo/puño - más abajo (Y menos negativa = más abajo)
        self.add_sphere(Sphere((-1.3, 0.1, -2.6), 0.3, body_material))
        
        # ============= BRAZO DERECHO (más visible en perspectiva 3/4) =============
        # Hombro/brazo superior - a la altura del torso
        self.add_sphere(Sphere((0.9, -0.2, -3.2), 0.4, body_material))
        # Antebrazo/puño - más abajo
        self.add_sphere(Sphere((1.4, 0.2, -3.4), 0.35, body_material))
        
        # ============= PIERNA IZQUIERDA =============
        # Muslo superior - ABAJO del torso (Y positiva = abajo)
        self.add_sphere(Sphere((-0.4, 0.6, -2.8), 0.4, body_material))
        # Pie/pantorrilla - MÁS ABAJO (Y más positiva = más abajo)
        self.add_sphere(Sphere((-0.6, 1.2, -2.6), 0.35, body_material))
        
        # ============= PIERNA DERECHA =============
        # Muslo superior - ABAJO del torso
        self.add_sphere(Sphere((0.4, 0.6, -3.2), 0.4, body_material))
        # Pie/pantorrilla - MÁS ABAJO
        self.add_sphere(Sphere((0.6, 1.2, -3.4), 0.35, body_material))
        
        # ============= OJOS =============
        eye_material = Material(
            diffuse=(0.9, 0.1, 0.1),
            specular=(1, 1, 1),
            ambient=(0.09, 0.01, 0.01),
            shininess=64,
            reflectivity=0.3
        )
        # Ojo izquierdo - en la parte FRONTAL de la cabeza
        self.add_sphere(Sphere((0.05, -1.0, -1.9), 0.08, eye_material))
        # Ojo derecho - en la parte FRONTAL de la cabeza
        self.add_sphere(Sphere((0.35, -1.0, -1.9), 0.08, eye_material))
        
        # ============= GARRAS BLANCAS =============
        claw_material = Material(
            diffuse=(0.9, 0.9, 0.8),
            specular=(1, 1, 1),
            ambient=(0.09, 0.09, 0.08),
            shininess=32,
            reflectivity=0.1
        )
        
        # Garras pie izquierdo (3 garras) - en la parte MÁS BAJA (Y más positiva)
        self.add_sphere(Sphere((-0.75, 1.4, -2.3), 0.06, claw_material))
        self.add_sphere(Sphere((-0.55, 1.4, -2.2), 0.06, claw_material))
        self.add_sphere(Sphere((-0.45, 1.4, -2.4), 0.06, claw_material))
        
        # Garras pie derecho (3 garras) - en la parte MÁS BAJA
        self.add_sphere(Sphere((0.75, 1.4, -3.1), 0.06, claw_material))
        self.add_sphere(Sphere((0.55, 1.4, -3.2), 0.06, claw_material))
        self.add_sphere(Sphere((0.45, 1.4, -3.5), 0.06, claw_material))
        
        # ============= ILUMINACIÓN =============
        
        # Luz principal desde arriba-derecha (Y negativa = arriba)
        self.add_light(Light((2, -2, 0), (1, 1, 1), 1.2))
        
        # Luz de relleno desde la izquierda
        self.add_light(Light((-1.5, -1, -1), (0.8, 0.8, 1), 0.6))
        
        # Luz trasera para definir la silueta
        self.add_light(Light((0, 0, -6), (0.9, 0.7, 0.5), 0.4))
        
        # Luz ambiente suave desde abajo (Y positiva = abajo)
        self.add_light(Light((0, 1, -2), (0.7, 0.7, 0.8), 0.3))
        
        material_names = ["Roca", "Metal", "Agua"]
        print(f"¡Golem creado con orientación CORREGIDA!")
        print(f"Material actual: {material_names[self.current_material_type]}")
        print("Estructura (con coordenadas Y invertidas):")
        print("- Cabeza: Y = -0.9 (arriba)")
        print("- Ojos: Y = -1.0 (en la cabeza)")
        print("- Torso: Y = -0.1 (centro)")
        print("- Brazos: Y = -0.3 a 0.2 (del torso hacia abajo)")
        print("- Piernas: Y = 0.6 a 1.2 (abajo)")
        print("- Garras: Y = 1.4 (parte más baja)")
        print("Usa las teclas 1, 2, 3 para cambiar materiales")
    
    def save_bmp(self, filename):
        file_size = 54 + 3*self.width*self.height
        header = bytearray(54)
        header[0:2] = b'BM'
        header[2:6] = file_size.to_bytes(4,'little')
        header[10:14] = (54).to_bytes(4,'little')
        header[14:18] = (40).to_bytes(4,'little')
        header[18:22] = self.width.to_bytes(4,'little')
        header[22:26] = self.height.to_bytes(4,'little')
        header[26:28] = (1).to_bytes(2,'little')
        header[28:30] = (24).to_bytes(2,'little')

        pixel_data = bytearray()
        for y in range(self.height):
            for x in range(self.width):
                c = self.frameBuffer[x][y]
                b = int(c[2]*255); g = int(c[1]*255); r = int(c[0]*255)
                pixel_data.extend([b,g,r])
            pad = (4 - (self.width*3)%4)%4
            pixel_data.extend([0]*pad)

        with open(filename,'wb') as f:
            f.write(header)
            f.write(pixel_data)
        print(f"Archivo BMP guardado: {filename}")