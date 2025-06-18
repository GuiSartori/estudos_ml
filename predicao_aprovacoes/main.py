import pandas as pd 
from sklearn import tree
import matplotlib.pyplot as plt

# Define o caminho do arquivo CSV
file_path = r'raw_data\full_dataset.csv'

# Read the CSV file
df = pd.read_csv(file_path)


# Define as colunas de entrada (features) e a coluna alvo (target) para o modelo
features = ['Horas_Estudo_Semanal', 'Faltas_Mes', 'Nota_Simulado']
target = ['Resultado']

# Separa os dados de entrada (x) e saída (y) para o modelo
x = df[features]
y = df[target]

# %%

# Cria o classificador de árvore de decisão com profundidade máxima 3 e semente aleatória 42
model = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
# Treina o modelo com os dados de entrada (x) e saída (y)
model.fit(x, y)

plt.figure(figsize=(16, 8), dpi=120)  # Ajuste os valores conforme necessário
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
plt.show()

# --------------------------------------------------------------------
