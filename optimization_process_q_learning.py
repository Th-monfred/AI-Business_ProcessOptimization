# Otimização fluxos de armazém com Q-learing

# importação de bibliotecas
import numpy as np

#Configuração dos parâmetros gamma e alpha para o Q-Learingn
gamma = 0.75
alpha = 0.9

# Parte 1 - Definição do ambiente
#dicionario python: estados locais
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}
#ações
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

R = np.array([
        [0,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0],
        [0,1,0,0,0,0,0,0,0,1,0,0],
        [0,0,1,0,0,0,1000,1,0,0,0,0],
        [0,0,0,1,0,0,1,0,0,0,0,1],
        [0,0,0,0,1,0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,1],
        [0,0,0,0,0,0,0,1,0,0,1,0],
       ])

#Parte 2 - Construção da solução de IA com Q-Learning
# Inicialização de valores Q
Q = np.array(np.zeros([12,12]))

#1) we select a random state st from our 12 possible states
for i in range(1000):
    current_state = np.random.randint(0,12)
#2) We play a random action at that can lead to a next possible state, i.e. such R(st, at) > 0
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
#3) We reach the next state st+1 and we get the reward R(st, at)
#4) We compute the Temporal Difference TDt(st,at):    
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]    
#5) We update the Q-value by applying the Bellman equation:    
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD
    
    
    
#Parte 3 - Resultados (Entrando em produção)
    
state_to_location = {state: location for location, state in location_to_state.items()}

def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# impressão da rota final
    
print('Route: ')
route('K', 'G')