import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Configuração da página
st.set_page_config(
    page_title="Sommelier Digital: Te ajudando a escolher o melhor vinho!",
    page_icon="🍷",
    layout="wide"
)

# --- CABEÇALHO ---
st.title("🍷 Sommelier Digital")
st.markdown("""
*Um guia para não cair em armadilhas na adega.*  
*Porque vida é muito curta para tomar um vinho ruim...*
""")

# --- SEÇÃO 1: EXPLICAÇÃO DO PROBLEMA ---
with st.expander("🔍 Por que usar Machine Learning para escolher vinho?", expanded=True):
    st.markdown("""
    ### O Desafio  
    Nem todo vinho com rótulo bonito é bom! Este modelo prevê se um vinho é bom (nota > 6.5) ou ruim (nota ≤ 6.5) com base em características químicas que você encontra no rótulo!  

    ### O Dataset  
    Foram usados dados de vinhos tintos e brancos com:  
    - 12 variáveis químicas (acidez, açúcar residual, pH, etc.).  
    - Avaliações de especialistas (0 a 10).  
    - Target: `vinho_bom` (0 ou 1, baseado na nota).  
    """)
    

# --- SEÇÃO 2: PROCESSO E JUSTIFICATIVAS ---
with st.expander("⚙️ Como a Mágica Acontece:", expanded=False):
    st.markdown("""
    ### Pré-Processamento: Limpeza e Padronização  
    1. Variáveis Categóricas:  
       - `type` (tinto/branco) virou `type_white` (0 ou 1).  
    2. Padronização:  
       - Todas as features numéricas foram escalonadas para média 0 e desvio padrão 1.  
    3. Target:  
       - Notas foram transformadas em binário: 0 (≤6.5) ou 1 (>6.5).  

    ### Por Que Extra Trees?  
    - Alta Acurácia (88.8%) e AUC (0.92): Melhor que outros modelos testados.  
    - Robustez: Menos propenso a overfitting.  
    - Velocidade: Rápido para deploy em apps.  
    """)

# --- SEÇÃO 3: INTERATIVIDADE ---
st.header("🎯 Teste Você Mesmo!")
st.markdown("""
*Preencha as informações do rótulo ao lado e descubra se o vinho é bom ou um desastre total.*  

""")

with st.sidebar:
    st.header("🧾 Detalhes do Rótulo")
    tipo_vinho = st.radio("Tipo do Vinho", ["Tinto", "Branco"], index=0, help="Tinto ou Branco?")
    alcool = st.slider("Teor Alcoólico (% ABV)", 5.0, 15.0, 12.5, 0.1, help="Quanto maior, mais encorpado!")
    acucar = st.slider("Açúcar Residual (g/dm³)", 0.0, 20.0, 2.5, 0.1, help="Seco (<4) ou Doce (>45)?")
    ph = st.slider("pH (acidez)", 2.8, 4.0, 3.4, 0.1, help="Quanto menor, mais ácido!")
    acidez = st.slider("Acidez Volátil (g/dm³)", 0.1, 1.0, 0.5, 0.01, help="Cuidado com valores >0.6")
    sulfatos = st.slider("Sulfatos (g/dm³)", 0.3, 1.5, 0.8, 0.1, help="Conservantes: mais = maior longevidade.")

modelo = load_model('modelo_vinho_completo2') 

# Botão de predição
if st.button("🍾 Verificar Qualidade"):

    input_data = pd.DataFrame({
        'type_white': [1 if tipo_vinho == "Branco" else 0],
        'alcohol': [alcool],
        'residual sugar': [acucar],
        'pH': [ph],
        'volatile acidity': [acidez],
        'sulphates': [sulfatos]
    })
    
    # Fazer a predição
    prediction = predict_model(modelo, data=input_data)
    qualidade = prediction['prediction_label'][0]
    prob_bom = prediction['prediction_score_1'][0]  # Probabilidade de ser "bom"
    
    # Exibir resultado
    if qualidade == 1:
        st.balloons()
        st.success(f"**✅ BOM (probabilidade: {prob_bom*100:.1f}%)**\n\nEste vinho tem alta chance de ser aprovado por especialistas!")
    else:
        st.error(f"**❌ NÃO É BOM (probabilidade: {(1-prob_bom)*100:.1f}%)**\n\nMelhor deixar na prateleira...")
# --- RODAPÉ DIVERTIDO ---
st.markdown("---")
st.markdown("""
Feito com ❤️ por Beatriz Trindade.  
Dados reais, paixão por vinho e um pouco de Python.  
""")

