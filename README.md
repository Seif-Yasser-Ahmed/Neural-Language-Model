# Neural Language Model

A simple neural language model built with TensorFlow 2 and Keras. This project demonstrates how to preprocess raw text, turn it into training sequences, train an LSTM-based next-word predictor, and generate new text.

## Table of Contents

- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Dataset](#dataset)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Text Generation](#text-generation)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features

- Cleans and tokenizes raw text  
- Builds input/output sequences for next-word prediction  
- Configurable LSTM model (stacked LSTM, optional bidirectional)  
- Trains with TensorFlow 2 / Keras  
- Simple text generation from a seed phrase  

## Getting Started

### Prerequisites

- Python 3.7+  
- TensorFlow 2.x  
- NumPy  
- pandas  
- BeautifulSoup4  

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Seif-Yasser-Ahmed/Neural-Language-Model.git
   cd Neural-Language-Model
```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

By default, the notebook expects a plain-text file (for example, Plato’s *Republic*) at `data/input.txt`. You can substitute your own text:

1. Place your `.txt` file in `data/`, e.g. `data/republic_sequences.txt`.
2. Adjust the filename in the first code cell of `main.ipynb` if needed.

---

## Data Preprocessing

1. **Load** raw text from file.
2. **Clean** HTML or punctuation (using `BeautifulSoup` + simple regex).
3. **Tokenize** into words, build a vocabulary.
4. **Create sequences** of length *N*+1 (first *N* tokens as input, last token as label).
5. **Save** sequences to `republic_sequences.txt` for faster reload.

---

## Model Architecture

Built with Keras’ `Sequential` API:

1. **Embedding** layer (vocab\_size × embedding\_dim)
2. **LSTM** layer (e.g. 100 units, return\_sequences=True)
3. **LSTM** layer (100 units)
4. **Dense** layer (100 units, ReLU)
5. **Output** Dense layer (vocab\_size, softmax)

*You can swap in GRU, add dropout, or bidirectional wrappers as experiments.*

---

## Training

In the notebook:

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=50)
```

* **Batch size:** 128
* **Epochs:** configurable (start with 20–50)

Training checkpoints (model weights) can be saved via Keras callbacks.

---

## Text Generation

Given a seed text:

1. Tokenize and pad to the input sequence length.
2. Predict next word with `model.predict()`.
3. Append predicted word, shift window, repeat.

Example in notebook:

```python
generate_text(model, tokenizer, seq_length, seed_text='happy families are')
```

---

## Project Structure

```
.
├── data/
│   └── input.txt             # raw text source
├── notebooks/
│   └── neural-language-model.ipynb
├── republic_sequences.txt    # generated training sequences
├── requirements.txt
└── README.md
```

---

## Usage

1. Open the notebook:

   ```bash
   jupyter notebook notebooks/neural-language-model.ipynb
   ```
2. Follow the sections in order:

   * Data loading & cleaning
   * Sequence creation
   * Model definition
   * Training & evaluation
   * Text generation

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to:

* Try different model architectures (e.g., Transformers)
* Add dropout, regularization
* Improve text-cleaning pipeline
* Integrate more advanced sampling methods for generation

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

