import math
import random
import time

# Problema do Caixeiro-Viajante
# Felipe e Huan Sousa

# UTILITÁRIOS E MODELAGEM DE DADOS

def gerar_cidades(n, largura=1000, altura=1000, seed_mapa=42):
    """Gera coordenadas (x, y) determinísticas para o mapa de n cidades sem travar o random global."""
    rng = random.Random(seed_mapa)
    return [(rng.uniform(0, largura), rng.uniform(0, altura)) for _ in range(n)]

def calcular_matriz_distancias(cidades):
    """Pré-computa matriz de adjacência D[i][j] em O(n^2)."""
    n = len(cidades)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(cidades[i][0] - cidades[j][0], cidades[i][1] - cidades[j][1])
            matriz[i][j] = d
            matriz[j][i] = d
    return matriz

def calcular_custo_rota(rota, matriz_d):
    """Calcula o custo total do ciclo hamiltoniano em O(n)."""
    custo = 0.0
    n = len(rota)
    for i in range(n - 1):
        custo += matriz_d[rota[i]][rota[i + 1]]
    custo += matriz_d[rota[-1]][rota[0]]  # Aresta de retorno à origem
    return custo

# OPERADORES DE VIZINHANÇA E RECOMBINAÇÃO

def operador_2opt(rota):
    """Inverte o segmento entre os índices i e j."""
    n = len(rota)
    i, j = sorted(random.sample(range(n), 2))
    # Rota = [0...i-1] + invertido(i...j) + [j+1...n-1]
    nova_rota = rota[:i] + rota[i:j + 1][::-1] + rota[j + 1:]
    return nova_rota

def operador_swap(rota):
    """Troca duas cidades de posição."""
    nova_rota = list(rota)
    i, j = random.sample(range(len(rota)), 2)
    nova_rota[i], nova_rota[j] = nova_rota[j], nova_rota[i]
    return nova_rota

def order_crossover_ox(pai1, pai2):
    """Order Crossover (OX) para permutação: garante ciclos válidos."""
    n = len(pai1)
    i, j = sorted(random.sample(range(n), 2))
    
    filho1 = [None] * n
    filho2 = [None] * n
    
    # Copia o segmento intermediário
    filho1[i:j + 1] = pai1[i:j + 1]
    filho2[i:j + 1] = pai2[i:j + 1]
    
    # Preenche o restante preservando a ordem relativa circular
    def preencher(filho, doador):
        idx_filho = (j + 1) % n
        idx_doador = (j + 1) % n
        conjunto_existente = set(filho[i:j + 1])
        
        while None in filho:
            cidade = doador[idx_doador]
            if cidade not in conjunto_existente:
                filho[idx_filho] = cidade
                idx_filho = (idx_filho + 1) % n
            idx_doador = (idx_doador + 1) % n

    preencher(filho1, pai2)
    preencher(filho2, pai1)
    
    return filho1, filho2

# TÊMPERA SIMULADA (SIMULATED ANNEALING)

def tempera_simulada(matriz_d, t_inicial=1000.0, t_min=1e-3, alfa=0.99, iter_por_temp=100):
    n = len(matriz_d)
    rota_atual = list(range(n))
    random.shuffle(rota_atual)
    
    custo_atual = calcular_custo_rota(rota_atual, matriz_d)
    melhor_rota = list(rota_atual)
    melhor_custo = custo_atual
    
    T = t_inicial
    
    while T > t_min:
        for _ in range(iter_por_temp):
            # Gera vizinho com 2-opt
            vizinho = operador_2opt(rota_atual)
            custo_vizinho = calcular_custo_rota(vizinho, matriz_d)
            
            delta = custo_vizinho - custo_atual
            
            # Critério de Metrópolis
            if delta < 0 or random.random() < math.exp(-delta / T):
                rota_atual = vizinho
                custo_atual = custo_vizinho
                
                if custo_atual < melhor_custo:
                    melhor_rota = list(rota_atual)
                    melhor_custo = custo_atual
                    
        T *= alfa  # Decaimento geométrico
        
    return melhor_rota, melhor_custo

# ALGORITMO GENÉTICO (AG)

def selecao_torneio(populacao, custos, k=3):
    indices = random.sample(range(len(populacao)), k)
    melhor_idx = min(indices, key=lambda idx: custos[idx])
    return populacao[melhor_idx]

def algoritmo_genetico(matriz_d, tam_pop=100, geracoes=300, p_crossover=0.85, p_mutacao=0.03, k_torneio=3):
    n = len(matriz_d)
    
    # Inicializa população aleatória
    populacao = []
    for _ in range(tam_pop):
        ind = list(range(n))
        random.shuffle(ind)
        populacao.append(ind)
        
    custos = [calcular_custo_rota(ind, matriz_d) for ind in populacao]
    
    idx_melhor = custos.index(min(custos))
    melhor_global = list(populacao[idx_melhor])
    melhor_custo_global = custos[idx_melhor]
    
    for _ in range(geracoes):
        nova_populacao = [list(melhor_global)]  # Elitismo: preserva o melhor
        
        while len(nova_populacao) < tam_pop:
            pai1 = selecao_torneio(populacao, custos, k_torneio)
            pai2 = selecao_torneio(populacao, custos, k_torneio)
            
            if random.random() < p_crossover:
                f1, f2 = order_crossover_ox(pai1, pai2)
            else:
                f1, f2 = list(pai1), list(pai2)
                
            # Mutação
            if random.random() < p_mutacao:
                f1 = operador_swap(f1)
            if random.random() < p_mutacao:
                f2 = operador_swap(f2)
                
            nova_populacao.append(f1)
            if len(nova_populacao) < tam_pop:
                nova_populacao.append(f2)
                
        populacao = nova_populacao
        custos = [calcular_custo_rota(ind, matriz_d) for ind in populacao]
        
        idx_melhor_gen = custos.index(min(custos))
        if custos[idx_melhor_gen] < melhor_custo_global:
            melhor_custo_global = custos[idx_melhor_gen]
            melhor_global = list(populacao[idx_melhor_gen])
            
    return melhor_global, melhor_custo_global

# EXECUÇÃO COMPARATIVA DE TESTE (30 RODADAS INDEPENDENTES)

if __name__ == "__main__":
    tamanhos = [20, 50, 100]
    num_rodadas = 30
    
    print("=" * 86)
    print(f"{'Instância':<10} | {'Algoritmo':<18} | {'Dist. Média':<14} | {'Melhor Dist.':<14} | {'Tempo Médio (s)':<15}")
    print("=" * 86)
    
    for n in tamanhos:
        # Gera o mesmo mapa fixo para as 30 rodadas sem travar o gerador randômico dos algoritmos
        cidades = gerar_cidades(n, seed_mapa=123)
        matriz_d = calcular_matriz_distancias(cidades)
        
        # 1. Têmpera Simulada (30 execuções)
        custos_sa = []
        tempos_sa = []
        for _ in range(num_rodadas):
            t0 = time.time()
            _, c = tempera_simulada(matriz_d, t_inicial=1000.0, t_min=0.01, alfa=0.95, iter_por_temp=50)
            tempos_sa.append(time.time() - t0)
            custos_sa.append(c)
            
        media_sa = sum(custos_sa) / num_rodadas
        melhor_sa = min(custos_sa)
        tempo_med_sa = sum(tempos_sa) / num_rodadas
        print(f"n = {n:<6} | {'Têmpera Simulada':<18} | {media_sa:<14.2f} | {melhor_sa:<14.2f} | {tempo_med_sa:<15.4f}")
        
        # 2. Algoritmo Genético (30 execuções)
        custos_ag = []
        tempos_ag = []
        for _ in range(num_rodadas):
            t0 = time.time()
            _, c = algoritmo_genetico(matriz_d, tam_pop=60, geracoes=200, p_crossover=0.85, p_mutacao=0.05)
            tempos_ag.append(time.time() - t0)
            custos_ag.append(c)
            
        media_ag = sum(custos_ag) / num_rodadas
        melhor_ag = min(custos_ag)
        tempo_med_ag = sum(tempos_ag) / num_rodadas
        print(f"n = {n:<6} | {'Algoritmo Genético':<18} | {media_ag:<14.2f} | {melhor_ag:<14.2f} | {tempo_med_ag:<15.4f}")
        print("-" * 86)
