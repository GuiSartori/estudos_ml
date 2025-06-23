# %%

import pandas as pd 
from sklearn import tree
import matplotlib.pyplot as plt

# %%

# Define o caminho do arquivo CSV
file_path = r'D:\VSCode\ML\predicao_aprovacoes\raw_data\full_dataset.csv'

# Read the CSV file
df = pd.read_csv(file_path)
df

# %%
# Cria o classificador de árvore de decisão com profundidade máxima 3 e semente aleatória 42
arvore = tree.DecisionTreeClassifier(max_depth=3, random_state=42)

# Define as colunas de entrada (features) e a coluna alvo (target) para o modelo
features = ['Horas_Estudo_Semanal', 'Faltas_Mes', 'Nota_Simulado']
target = 'Resultado'

# Separa os dados de entrada (x) e saída (y) para o modelo
x = df[features]
y = df[target]

# %%
# Treina o modelo com os dados de entrada (x) e saída (y)
arvore.fit(x, y)

# %%
arvore.predict([[5,5,5]])

# %%
arvore.predict_proba([[5,5,5]])

# %%
# Obtém as probabilidades de cada classe para a entrada especificada
probabilidades = arvore.predict_proba([[5,5,5]])[0]
pd.Series(probabilidades, index=arvore.classes_)

# %%
plt.figure(dpi=400, figsize=(8, 8))  # Ajuste os valores conforme necessário
tree.plot_tree(arvore,
                feature_names=features,
                class_names=arvore.classes_,
                filled=True)
plt.show()


# %%
