# Drug Interaction Analysis System

A comprehensive system for analyzing drug interactions using biomedical language models and network analysis.

## Overview

This project implements an advanced drug interaction analysis system that combines multiple approaches to provide comprehensive drug interaction information:

1. **Biomedical Language Model Integration**: Utilizes state-of-the-art biomedical language models (BioMedLM) to analyze drug interactions and extract information from medical literature.
2. **Drug Interaction Database**: Maintains a curated database of known drug interactions with severity levels and evidence sources.
3. **Network Analysis**: Visualizes drug interaction networks to help understand complex medication relationships.
4. **Clinical Notes Analysis**: Extracts medication information and potential interactions from clinical notes.

## Key Features

### 1. Drug Interaction Analysis
- Query specific drug pairs for interaction information
- Get severity levels (Mild, Moderate, Severe)
- Access evidence sources and management recommendations
- View interaction networks and visualizations

### 2. Drug Information Retrieval
- Comprehensive drug information including:
  - Drug class
  - Mechanism of action
  - Common indications
  - Side effects
  - Common interactions
  - Contraindications

### 3. Clinical Notes Analysis
- Extract medications from clinical notes
- Identify potential drug interactions
- Analyze medication patterns
- Generate structured reports

### 4. Interactive Visualization
- Network graphs of drug interactions
- Color-coded severity indicators
- Interactive exploration of drug relationships
- Multi-level interaction depth analysis

## Technical Implementation

### Core Components

1. **BiomedicalLLM Class**
   - Handles biomedical language model inference
   - Processes natural language queries
   - Extracts structured information from medical literature
   - Provides fallback mechanisms for robustness

2. **DrugInteractionDatabase Class**
   - Maintains curated drug interaction data
   - Handles drug name normalization and aliases
   - Implements semantic search for drug names
   - Provides interaction lookup functionality

3. **DDIProcessor Class**
   - Processes drug interaction queries
   - Extracts drug names from natural language
   - Analyzes clinical notes
   - Generates interaction networks

4. **DrugInteractionChatbot Class**
   - Provides natural language interface
   - Handles user queries and responses
   - Integrates all system components
   - Formats responses for readability

### Web Interface

The system includes a Flask-based web application with:
- Interactive chat interface
- Real-time drug interaction visualization
- Clinical notes analysis
- Drug information display
- Responsive design using Tailwind CSS

## Usage

### Command Line Interface
```python
from hi import DrugInteractionChatbot

# Initialize the chatbot
chatbot = DrugInteractionChatbot()

# Process queries
response = chatbot.process_message("Can I take aspirin and warfarin together?")
print(response)
```

### Web Interface
```bash
# Run the web application
python hi.py web
```

### Clinical Notes Analysis
```python
# Analyze clinical notes
clinical_text = "Patient is taking metformin 500mg twice daily and warfarin 5mg daily..."
results = chatbot.processor.extract_drugs_from_clinical_notes(clinical_text)
```

## Dependencies

- Python 3.7+
- PyTorch
- Transformers
- NetworkX
- Matplotlib
- Flask
- Sentence Transformers
- scikit-learn
- pandas
- numpy

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford CRFM for BioMedLM
- The biomedical research community for drug interaction data
- Open source community for various tools and libraries used in this project 