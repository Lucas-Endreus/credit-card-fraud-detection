# 🛡️ Credit Card Fraud Detection using Machine Learning

> Projeto desenvolvido para identificar transações fraudulentas utilizando técnicas de Machine Learning e análise exploratória de dados.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-black?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📖 Sobre o projeto

A fraude em cartões de crédito representa um dos maiores desafios do setor financeiro.

Mesmo que a quantidade de fraudes seja muito pequena quando comparada ao volume total de transações, o impacto financeiro é extremamente elevado.

Este projeto apresenta uma solução completa utilizando Machine Learning para identificar transações fraudulentas, comparando diferentes algoritmos e avaliando seu desempenho através de métricas adequadas para problemas de classificação desbalanceada.

---

# 🎯 Objetivos

- Desenvolver um pipeline completo de Machine Learning.
- Realizar análise exploratória dos dados (EDA).
- Aplicar Feature Engineering.
- Comparar diferentes algoritmos de classificação.
- Avaliar os modelos utilizando métricas apropriadas.
- Documentar todo o processo de desenvolvimento.

---

# 📂 Estrutura do Projeto

```text
credit-card-fraud-detection/

├── assets/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── main.py
```

---

# 🧰 Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib

---

# 📊 Dataset

O conjunto de dados utilizado é público e contém aproximadamente **285 mil transações financeiras**, sendo apenas uma pequena parcela classificada como fraude.

Esse forte desbalanceamento torna o problema particularmente desafiador e exige métricas mais robustas do que apenas a acurácia.

---

# 🔎 Etapas do Projeto

## 1. Análise Exploratória (EDA)

Foram realizadas análises para:

- distribuição das classes;
- identificação de valores nulos;
- detecção de duplicatas;
- análise estatística;
- histogramas;
- boxplots;
- heatmap de correlação.

---

## 2. Engenharia de Atributos

As principais transformações realizadas foram:

- criação da variável `Hour`;
- transformação logarítmica da variável `Amount`;
- preparação dos dados para modelagem.

---

## 3. Modelagem

Os seguintes modelos foram treinados:

- Logistic Regression
- Random Forest
- XGBoost

---

## 4. Avaliação

Os modelos foram avaliados utilizando:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Além disso, foi realizado o ajuste de diferentes thresholds para analisar o impacto entre precisão e recall.

---

# 📈 Resultados

Durante os experimentos, o modelo **Random Forest** apresentou o melhor equilíbrio entre Precision, Recall e F1-score no threshold padrão de 0,5.

O modelo **XGBoost** apresentou o melhor desempenho na métrica PR-AUC, demonstrando excelente capacidade para problemas altamente desbalanceados.

## Resultados dos modelos

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9770 | 0.0625 | **0.8784** | 0.1167 | 0.9702 | 0.6990 |
| Random Forest | **0.9991** | **0.7329** | 0.7973 | **0.7638** | **0.9773** | 0.7764 |
| XGBoost | 0.9987 | 0.5837 | 0.8243 | 0.6835 | 0.9673 | **0.8066** |
---
No threshold padrão de `0.50`, o Random Forest apresentou o melhor equilíbrio geral.  
No ajuste de threshold, o Random Forest atingiu seu maior F1-score em `0.70`:

- Precision: 84,38%
- Recall: 72,97%
- F1-score: 78,26%

# 📷 Exemplos de Resultados

## Comparação entre modelos

> Adicione aqui a imagem:
## Comparação entre modelos

![Comparação entre modelos](assets/model_comparison.png)

---

## Curva ROC

![Curva ROC](assets/roc_curve.png)

---

## Curva Precision-Recall

![Curva Precision-Recall](assets/precision_recall_curve.png)

# 🚀 Como executar

Clone o repositório:

```bash
git clone https://github.com/Lucas-Endreus/credit-card-fraud-detection.git
```

Entre na pasta:

```bash
cd credit-card-fraud-detection
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente:

Windows

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

---

# 📊 Pipeline

O pipeline desenvolvido realiza automaticamente:

- carregamento do dataset;
- preparação dos dados;
- treinamento dos modelos;
- avaliação;
- comparação de métricas;
- geração dos gráficos;
- salvamento dos resultados.

---

# 📌 Principais Aprendizados

Durante o desenvolvimento deste projeto foi possível aprofundar conhecimentos em:

- Ciência de Dados;
- Machine Learning;
- Feature Engineering;
- Avaliação de Modelos;
- Organização de projetos em Python;
- Estruturação de pipelines reutilizáveis.

---

# 🔮 Melhorias Futuras

- Deploy utilizando FastAPI.
- Interface Web com Streamlit.
- Otimização automática de hiperparâmetros.
- Integração com banco de dados.
- Monitoramento do desempenho do modelo.

---

# 👨‍💻 Autor

**Lucas Frota**

Projeto desenvolvido como parte da formação em Inteligência Artificial e Ciência de Dados da DIO.

---

# 📄 Licença

Este projeto está licenciado sob a licença MIT.
