<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=FFC905&height=180&section=header&animation=twinkling"/>

# 🐶🐱 Classificador de Imagens: Cães vs Gatos

Este projeto utiliza **Visão Computacional e Deep Learning com Keras/TensorFlow** para classificar imagens como **gato** ou **cachorro**. Além do modelo treinado, a aplicação conta com uma interface gráfica em Python que permite ao usuário carregar uma imagem e obter o resultado da classificação de forma visual e intuitiva.

---

## ⚙️ Como executar o projeto do zero

### 1. ✅ Criar ambiente virtual Python no VS Code

```bash
# Crie o ambiente
python -m venv .venv

# Ative o ambiente virtual
# No Windows:
.venv\Scripts\activate

# No Linux/macOS:
source .venv/bin/activate
```

> No VS Code, selecione o kernel Python do ambiente virtual:  
> `Ctrl+Shift+P` → `Python: Select Interpreter` → escolha o `.venv`.

---

### 2. 📦 Instalar as dependências

```bash
pip install -r requisitos.txt
```

Ou instale manualmente:
```bash
pip install numpy opencv-python tensorflow pillow tqdm matplotlib
```

---

### 3. 📁 Estrutura recomendada do projeto

```
projeto_caes_gatos/
│
├── data/
│   └── train/
│       ├── cat/
│       └── dog/
├── modelo_caes_gatos.keras     ← gerado após o treinamento
├── treino_modelo.py            ← script de treinamento
├── interface_visual.py        ← interface gráfica
├── requisitos.txt
└── README.md
```

---

## 🚀 Como executar o projeto completo

### 🧪 1. Treinando o modelo

```bash
# Ative seu ambiente virtual
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/macOS

# Execute o script de treino
python treino_modelo.py
```

O modelo será salvo como `modelo_caes_gatos.keras`.

---

### 🖼️ 2. Usando a interface gráfica

Após treinar e salvar o modelo, execute a interface:

```bash
python interface_visual.py
```

- Um painel será aberto.
- Clique em **“Selecionar Imagem”**.
- Escolha uma imagem de gato ou cachorro do seu computador.
- O programa exibirá a imagem e a previsão.

---

## 🎯 Tema e Objetivo

**Tema:** Visão Computacional com Deep Learning  
**Objetivo:** Criar um classificador binário de imagens para distinguir **cães** e **gatos**, utilizando redes neurais convolucionais.

---

## 🔍 Funcionamento do Projeto

### 🧪 1. Pré-processamento dos dados
- Lê imagens com `OpenCV` (`cv2`)
- Redimensiona para 100x100 pixels
- Converte para arrays NumPy
- Normaliza (valores de 0 a 1)
- Associa rótulo: `0 = gato`, `1 = cachorro`

### 🧠 2. Rede Neural Convolucional (CNN)

```python
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

#### Estrutura:
| Camada            | Função |
|-------------------|--------|
| Conv2D (2x)       | Detecta padrões (bordas, texturas) |
| MaxPooling2D (2x) | Reduz o tamanho, mantendo os padrões |
| Flatten           | Transforma a imagem em vetor 1D |
| Dense             | Aprende combinações dos padrões |
| Dense (1)         | Saída binária com `sigmoid` |

### ⚙️ 3. Compilação e Treinamento

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, validation_split=0.2)
```

### 💾 4. Salvamento e Interface

```python
model.save("modelo_caes_gatos.keras")
```

A interface criada com `Tkinter` permite ao usuário selecionar uma imagem `.jpg` e visualizar:
- A imagem carregada
- A previsão do modelo (gato 🐱 ou cachorro 🐶)

---

## ✅ Pontos Fortes

- Completo: da preparação ao uso do modelo
- Fácil de adaptar para outras classificações de imagem
- Interface amigável
- Didático e modular

---

## 📉 Pontos a Melhorar

| Limitação               | Oportunidade de melhoria |
|------------------------|--------------------------|
| Dataset limitado       | Usar mais dados reais ou aplicar *data augmentation* |
| Modelo simples         | Adicionar mais camadas ou dropout |
| GUI básica             | Modernizar com PyQt5, CustomTkinter ou web (Flask) |
| Avaliação limitada     | Adicionar métricas como F1, precisão, matriz de confusão |

---

## 🧭 Possíveis Extensões

- Classificação com webcam em tempo real
- Exportar para `.exe` ou webapp
- Uso de modelos pré-treinados como VGG16 ou MobileNet
- Aplicação para outros domínios: frutas, roupas, radiografias, etc.

---

## 📬 Contato

Caso tenha dúvidas, sugestões ou interesse em colaborar, entre em contato comigo:

- 📧 E-mail: [gusthyerris@gmail.com](mailto:gusthyerris@gmail.com)  
- 💼 LinkedIn: [linkedin.com/in/gustavo-thyerris](https://www.linkedin.com/in/gustavo-thyerris/)
