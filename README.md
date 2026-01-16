# 💰 FinanceInsight-NER

AI-powered Financial Document Analysis using FinBERT Named Entity Recognition (NER). Extract key financial metrics, company information, and balance sheet data from annual reports and 10-K filings automatically.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- 🔍 **Intelligent PDF Parsing** - Extract text from complex financial documents
- ⚡ **Dual-Mode Analysis**
  - **Regex Mode**: Fast, memory-efficient pattern matching (no model required)
  - **AI Mode**: Advanced NER with fine-tuned FinBERT (95% accuracy)
- 💼 **MD&A Extraction** - Automatically identify management discussion metrics
- 📊 **Balance Sheet Detection** - Extract assets, liabilities, and equity
- 🎨 **Interactive Dashboard** - Beautiful Streamlit UI with real-time results
- 📥 **CSV Export** - Download extracted data for further analysis
- 🧠 **Memory Optimized** - Handles large PDFs with intelligent chunking

---

## 🎯 What Gets Extracted

| Entity Type | Examples |
|-------------|----------|
| **Organizations** | Microsoft, Tesla, JPMorgan Chase, Infosys |
| **Metrics** | Revenue, Net Income, Operating Income, EPS, ROE |
| **Values** | $83.4 billion, ₹75,000 crore, 8.8 million subscribers |
| **Dates** | Q1 2023, FY 2024, Fiscal Year 2025 |
| **Financial Statements** | Total Assets, Total Liabilities, Stockholders' Equity |

---

## 📋 Prerequisites

- Python 3.8 or higher
- 2GB+ RAM (4GB+ recommended for AI mode)
- Modern web browser

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/nidhijat45/FinanceInsight-NER.git
cd FinanceInsight-NER
```

### 2. Set Up Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install CPU-only PyTorch (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📖 Usage Guide

### 🟢 Regex Mode (Recommended - No Model Needed)

**Works out of the box!**

1. Launch the app: `streamlit run app.py`
2. Upload a financial PDF (10-K, annual report, etc.)
3. **Keep "Use AI Analysis" unchecked**
4. Click "🚀 Analyze Document"
5. View results in interactive tabs
6. Download CSV for further analysis

**Advantages:**
- ⚡ Very fast (10-30 seconds)
- 💾 Low memory usage (~50 MB)
- ✅ No model download needed
- 📊 ~80% accuracy

---

### 🔵 AI Mode (Advanced - Requires Model)

**For maximum accuracy**

1. Download or train a FinBERT NER model
2. Place model files in `./MyFinBERT_Model/` folder
3. In the app sidebar, enter model path
4. Click "🔄 Load AI Model"
5. Upload PDF and **check "Use AI Analysis"**
6. Click Analyze

**Advantages:**
- 🎯 Higher accuracy (~95%)
- 🧠 Better context understanding
- 📝 Identifies complex entities
- 🔗 Relationship extraction

---

## 📁 Project Structure
```
FinanceInsight-NER/
│
├── app.py                     # Streamlit UI (main application)
├── backend.py                 # Core analysis logic
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
│
├── MyFinBERT_Model/          # Your trained model (optional, not in repo)
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
│
└── venv/                     # Virtual environment (git ignored)
```

---

## 🛠️ Configuration

### Memory Settings

Edit `backend.py` to adjust:
```python
# Line ~20
self.max_length = 256          # Token limit per chunk

# Line ~40
max_pages: int = 50            # Max PDF pages to process

# Line ~80
chunk_size: int = 1000         # Characters per chunk
```

### Model Path

Default: `./MyFinBERT_Model`

Change in sidebar if your model is elsewhere.

---

## 🐛 Troubleshooting

### Memory Error: "not enough memory to allocate"

**Solutions:**
1. ✅ Disable "Use AI Analysis" (use regex-only mode)
2. ✅ Reduce `max_pages` in backend.py
3. ✅ Close other applications
4. ✅ Process smaller PDFs

### Model Not Loading

**Solutions:**
1. ✅ Verify model path is correct
2. ✅ Check model folder contains all required files
3. ✅ Use regex-only mode (no model needed)

### "No text extracted from PDF"

**Solutions:**
1. ✅ PDF might be image-based (needs OCR)
2. ✅ Try a different PDF
3. ✅ Check if PDF is password-protected

---

## 📊 Performance Comparison

| Mode | Speed | Memory | Accuracy | Model Needed |
|------|-------|--------|----------|--------------|
| **Regex** | ⚡ 10-30s | 💾 ~50 MB | 80% | ❌ No |
| **AI** | 🐌 1-3 min | 💾 ~500 MB | 95% | ✅ Yes |

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Tech Stack

- **Streamlit** - Interactive web UI
- **Transformers** (Hugging Face) - FinBERT model
- **PyTorch** - Deep learning framework
- **PDFPlumber** - PDF text extraction
- **Pandas** - Data manipulation
- **Regex** - Pattern matching

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Hugging Face** - Transformers library
- **FinBERT** - Pre-trained financial BERT model
- **Streamlit** - Web framework
- **PDFPlumber** - PDF processing

---

## 📧 Contact

**Nidhi Jat**
- GitHub: [@nidhijat45](https://github.com/nidhijat45)
- Repository: [FinanceInsight-NER](https://github.com/nidhijat45/FinanceInsight-NER)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes**. Always verify critical financial data from official sources.

---

<div align="center">

**Made with ❤️ using Python, Streamlit, and FinBERT**

⭐ **Star this repo if you found it helpful!**

</div>
