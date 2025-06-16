import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

def crear_cromosoma():
    cromosoma = []
    for i in range(39):
        examen = random.randint(0, 3)  # Ahora 4 exámenes
        genes = [0, 0, 0, 0]
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma

def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': [], 'D': []} 
    examenes = ['A', 'B', 'C', 'D']  
    
    for i in range(39):
        idx = i * 4  # Cada alumno tiene 4 genes
        for j in range(4):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    
    return asignaciones

def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    
    # Penalización si no hay 10 alumnos por examen
    if any(len(asignaciones[ex]) != 10 for ex in ['A', 'B', 'C', 'D']):
        return -1000
    
    promedios = {}
    desviaciones = {}
    for examen in ['A', 'B', 'C', 'D']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
        desviaciones[examen] = np.std(notas_examen)
    
    penalizacion_variacion = sum([desviaciones[ex] for ex in ['A', 'B', 'C', 'D']])

    diversidad = np.std([promedios[ex] for ex in ['A', 'B', 'C', 'D']])
    
    return -(penalizacion_variacion - diversidad)


def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    
    alumno1 = random.randint(0, 38)
    alumno2 = random.randint(0, 38)
    
    idx1 = alumno1 * 4
    idx2 = alumno2 * 4
    
    examen1 = [i for i in range(4) if cromosoma_mutado[idx1 + i] == 1][0]
    examen2 = [i for i in range(4) if cromosoma_mutado[idx2 + i] == 1][0]
    
    if examen1 != examen2:
        cromosoma_mutado[idx1:idx1+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx1 + examen2] = 1
        
        cromosoma_mutado[idx2:idx2+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx2 + examen1] = 1
    
    return cromosoma_mutado

def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    fitness_evolution = []  # Lista para almacenar la evolución del fitness
    
    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        nueva_poblacion = []
        
        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])
        
        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(poblacion[:tam_poblacion//2])
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        
        # Guardar el mejor fitness en cada generación
        mejor_fitness = fitness_scores[0][1]
        fitness_evolution.append(mejor_fitness)
        
        if gen % 20 == 0:
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.4f}")
    
    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma, fitness_evolution

print("REPRESENTACIÓN BINARIA")
print("Problema: Distribuir 39 alumnos en 4 exámenes (A, B, C, D) de forma equitativa")
print("Cromosoma: 156 bits (39 alumnos × 4 bits cada uno)")  # 3 bits por 4 bits
print("Gen: [0,0,1,0] significa alumno asignado a examen C\n") 

mejor_solucion, fitness_evolution = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución final:")
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Alumnos: {[alumnos[i] for i in indices[:5]]}... (mostrando primeros 5)")

print("\nVerificación de equilibrio:")
promedios = []
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")

# Gráfica de la evolución del fitness por generación
plt.figure(figsize=(10, 6))
plt.plot(fitness_evolution, label='Fitness Evolución')
plt.title('Evolución del Fitness por Generación')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()

# Gráficas de histogramas de notas por examen
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.hist(notas_examen, bins=10, edgecolor='black')
    plt.title(f'Histograma de Notas para Examen {examen}')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()


