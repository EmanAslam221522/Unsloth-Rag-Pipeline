

## 📖 Project Overview
This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for answering medical-related questions. The system retrieves relevant medical context from a vector database and generates accurate, context-grounded responses using a **4-bit quantized Large Language Model (LLM)** powered by **Unsloth**.

The project is optimized for **low-GPU environments** and runs efficiently on **Google Colab (T4 GPU)** with reduced memory usage.

> ⚠️ **Legal Disclaimer**  
> This project is strictly for **educational and research purposes only**.  
> It does **not provide medical advice**, diagnosis, or treatment.

---

## 🎯 Objectives
- Implement a complete RAG pipeline (Retrieve → Augment → Generate)
- Use **Unsloth dynamic 4-bit quantization** for memory efficiency
- Store and retrieve documents using a **vector database**
- Reduce hallucinations by grounding responses in retrieved context
- Ensure compatibility with free-tier GPU environments

---

## 🏗️ System Architecture
```

User Query
↓
Sentence Embedding
↓
Vector Database (ChromaDB)
↓
Relevant Medical Context
↓
Prompt Augmentation
↓
4-bit Quantized LLM (Unsloth)
↓
Context-Grounded Answer

```

---

## ✨ Key Features
- ✅ Complete Retrieval-Augmented Generation (RAG) pipeline  
- ✅ Unsloth-powered **4-bit quantized LLM**  
- ✅ Semantic search using **ChromaDB**  
- ✅ Medical domain document processing  
- ✅ Reduced GPU memory usage  
- ✅ Modular and easy-to-extend design  

---

## 📁 Project Structure
```

medical-rag-unsloth/
├── README.md
├── rag_pipeline.py
├── requirements.txt
├── medical_rag_colab.ipynb
└── .gitignore

````

---

## 🔧 Technical Implementation

### 1️⃣ Model Loading (Unsloth 4-bit Quantization)
The language model is loaded using Unsloth with dynamic 4-bit quantization to significantly reduce memory usage.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True
)
````

---

### 2️⃣ Vector Database (ChromaDB)

Medical documents are embedded using Sentence Transformers and stored in an **in-memory vector database** for semantic retrieval.

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("medical_knowledge")
```

---

### 3️⃣ Retrieval-Augmented Generation Flow

1. Convert user query into embeddings
2. Retrieve top relevant documents from vector database
3. Combine retrieved context with the user question
4. Generate response using the LLM

```text
Context + Question → Prompt → LLM → Answer
```

---

## 🚀 How to Run the Project

### ▶️ Google Colab (Recommended)

```bash
git clone https://github.com/YOUR-USERNAME/medical-rag-unsloth.git
cd medical-rag-unsloth
pip install -r requirements.txt
python rag_pipeline.py
```

### ▶️ Local Machine

```bash
pip install -r requirements.txt
python rag_pipeline.py
```

---

## 🧪 Example Usage

```python
from rag_pipeline import MedicalRAG

rag = MedicalRAG()

rag.add_documents([
    "Diabetes treatment includes Metformin 500mg twice daily.",
    "Hypertension target blood pressure is below 130/80 mmHg."
])

response = rag.query("What is the treatment for diabetes?")
print(response)
```

---

## 📊 Performance & Evaluation

* **Model Size:** ~1GB (4-bit quantized)
* **VRAM Usage:** ~5–7GB
* **Retrieval Time:** <0.5 seconds (small datasets)
* **Evaluation Type:** Qualitative evaluation based on context relevance

> Note: RAG systems are evaluated qualitatively unless benchmark datasets and metrics are explicitly defined.

---

## 🛠️ Technologies Used

* Python
* Unsloth
* Transformers
* PyTorch
* ChromaDB
* Sentence-Transformers
* NumPy

---

## 🎓 Learning Outcomes

* Understanding Retrieval-Augmented Generation (RAG)
* Implementing vector-based semantic search
* Deploying memory-efficient LLMs using 4-bit quantization
* Handling medical NLP responsibly
* Working with GPU constraints

