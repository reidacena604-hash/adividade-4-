import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da Página
st.set_page_config(page_title="Preditor de Notas - Linear Regression", layout="centered")

st.title("📊 Predição de Desempenho Acadêmico")
st.markdown("""
Este script utiliza a biblioteca `scikit-learn` para realizar uma **Regressão Linear Simples**, 
modelando a relação entre horas de estudo e a nota final obtida.
""")

# 1. Conjunto de Dados
data = {
    'horas': [2, 4, 5, 7, 9, 10],
    'notas': [1, 2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)

# 2. Treinamento do Modelo
X = df[['horas']]  # Feature (Independente)
y = df['notas']    # Target (Dependente)

model = LinearRegression()
model.fit(X, y)

# 3. Interface de Interação (Sidebar)
st.sidebar.header("Parâmetros de Entrada")
input_horas = st.sidebar.slider("Horas de Estudo:", min_value=0, max_value=15, value=6)

# Predição em tempo real
nota_prevista = model.predict([[input_horas]])[0]

# 4. Exibição de Resultados
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Horas Inseridas", value=f"{input_horas}h")
with col2:
    st.metric(label="Nota Prevista", value=f"{nota_prevista:.2f}")

# 5. Representação Gráfica
fig, ax = plt.subplots()
ax.scatter(df['horas'], df['notas'], color='blue', label='Dados Reais (Treino)')
ax.plot(df['horas'], model.predict(X), color='red', linestyle='--', label='Linha de Regressão')
ax.scatter([input_horas], [nota_prevista], color='green', marker='x', s=100, label='Sua Predição')

ax.set_xlabel("Horas de Estudo")
ax.set_ylabel("Nota")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# 6. Detalhes Técnicos (Expander)
with st.expander("Ver detalhes do modelo"):
    st.write(f"**Coeficiente (Slope):** {model.coef_[0]:.4f}")
    st.write(f"**Intercepto:** {model.intercept_:.4f}")
    st.write(f"**Equação:** Nota = ({model.coef_[0]:.2f} * Horas) + ({model.intercept_:.2f})")