# Math Education Assistant - Fine-tuning LLM with LoRA

##  Project Overview

This project implements a domain-specific AI assistant for mathematics education by fine-tuning a Large Language Model using parameter-efficient techniques. The assistant can answer math questions, explain concepts, and solve problems across various mathematical domains.

### Key Features
- Fine-tuned on 3,000+ math instruction-response pairs
- Uses LoRA (Low-Rank Adaptation) for efficient training on Google Colab free tier
- Comprehensive evaluation with ROUGE, BLEU, and Perplexity metrics
- Multiple hyperparameter experiments documented
- Interactive Gradio web interface for deployment
- Comparison between base and fine-tuned models

---

## Project Requirements Met

### 1. **Dataset** 
- **Source**: UltraData-Math from Hugging Face (`openbmb/UltraData-Math`)
- **Size**: 3,000 high-quality instruction-response pairs (adjustable)
- **Format**: Structured as math questions and step-by-step solutions
- **Preprocessing**: 
  - Tokenization with padding and truncation
  - Formatted into instruction-response templates
  - Split into 90% train / 10% validation

### 2. **Model Selection** 
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Why**: Efficient for Colab free GPU (T4), balances capability with practical constraints
- **Alternatives**: Gemma-2B, Phi-2 (can be easily swapped)
- **Fine-tuning Method**: LoRA via `peft` library
- **Optimization**: 4-bit quantization for memory efficiency

### 3. **LoRA Configuration** 
```python
LoraConfig(
    r=16,                    
    lora_alpha=32,          
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
```
- **Trainable Parameters**: ~4.2M (0.38% of total model)
- **Memory Efficient**: Fits on Colab free tier GPU

### 4. **Hyperparameter Experiments** 

Three experiments conducted with documented results:

| Experiment | Learning Rate | Batch Size | Grad Accum | Epochs | LoRA Rank | Notes |
|-----------|---------------|------------|------------|--------|-----------|-------|
| Exp1_Baseline | 2e-4 | 4 | 4 | 3 | 16 | Baseline configuration |
| Exp2_LowerLR | 1e-4 | 4 | 4 | 3 | 16 | Lower LR for stability |
| Exp3_HigherRank | 2e-4 | 4 | 4 | 3 | 32 | Higher capacity |

All experiments track:
- Final train/validation loss
- Training time (minutes)
- GPU memory usage (GB)

### 5. **Evaluation Metrics** 

The model is evaluated using multiple metrics:

- **ROUGE Scores** (1, 2, L): Measures overlap with reference answers
- **BLEU Score**: Evaluates translation-like quality
- **Perplexity**: Measures model confidence
- **Qualitative Testing**: 
  - In-domain math questions
  - Out-of-domain queries
  - Comparison with base model

### 6. **Deployment** 
- **Interface**: Gradio ChatInterface
- **Features**:
  - User-friendly chat UI
  - Example questions
  - Retry/Undo/Clear functionality
  - Shareable link (when share=True)

---

## Getting Started

### Prerequisites
- Google Colab account (free tier works!)
- Python 3.8+
- Internet connection for downloading models

### Installation

1. **Open the notebook in Google Colab**:
   - Upload `math_education_assistant.ipynb` to Colab
   - Or open directly from GitHub

2. **Set Runtime to GPU**:
   - Runtime → Change runtime type → T4 GPU (free tier)

3. **Run all cells sequentially**:
   - The first cell installs all dependencies
   - Subsequent cells execute the full pipeline

### Quick Start

```python
# The notebook is fully self-contained. Simply:
# 1. Open in Colab
# 2. Runtime → Run all
# 3. Wait for training to complete
# 4. Interact with the Gradio interface at the end
```

## Methodology

### 1. Data Preprocessing
```python
# Format: Instruction-Response pairs
### Instruction:
What is the Pythagorean theorem?

### Response:
The Pythagorean theorem states that in a right triangle, 
a² + b² = c², where c is the hypotenuse...
```

### 2. Training Pipeline
1. Load UltraData-Math dataset
2. Format into instruction-response pairs
3. Tokenize with max length 512
4. Apply LoRA to base model
5. Train with AdamW optimizer
6. Track metrics every 50 steps
7. Evaluate every 200 steps
8. Save best checkpoint

### 3. Evaluation Process
1. Load best model from experiments
2. Generate predictions on validation set
3. Calculate ROUGE, BLEU, Perplexity
4. Compare with base model responses
5. Test on out-of-domain queries

---

## Expected Results

### Training Efficiency
- **Training Time**: ~30-60 minutes per experiment (on Colab T4)
- **GPU Memory**: ~8-12 GB peak usage
- **Trainable Params**: ~4.2M (0.38% of total)

### Performance Metrics (Typical)
- **ROUGE-1**: 0.35-0.45
- **ROUGE-L**: 0.30-0.40
- **BLEU**: 15-25
- **Perplexity**: 8-15 (lower is better)

### Qualitative Improvements
- More accurate math explanations
- Better step-by-step problem solving
- Domain-appropriate terminology
- Structured responses

---

## Usage Examples

### In-Domain Queries (Math)
```
Q: What is the quadratic formula?
A: The quadratic formula is x = (-b ± √(b²-4ac)) / 2a, 
   used to solve equations of the form ax² + bx + c = 0...

Q: Solve for x: 2x + 5 = 15
A: Let's solve step by step:
   1. Subtract 5 from both sides: 2x = 10
   2. Divide by 2: x = 5
```

### Out-of-Domain Handling
```
Q: What's the weather like?
A: I'm a math education assistant focused on helping 
   with mathematical concepts and problem-solving. 
   I can't provide weather information...
```

---

## Customization

### Change Base Model
```python
# In the notebook, modify:
MODEL_NAME = "google/gemma-2b"  # Instead of TinyLlama
# Or use: "microsoft/phi-2"
```

### Adjust Dataset Size
```python
DATASET_SIZE = 5000  # Increase for better performance
# Trade-off: More data = longer training
```

### Modify LoRA Parameters
```python
lora_config = LoraConfig(
    r=32,              # Increase for more capacity
    lora_alpha=64,     # Scale accordingly
    # ...
)
```

### Add More Experiments
```python
# Add experiment 4, 5, etc.:
run_experiment(
    experiment_name="Exp4_CustomConfig",
    learning_rate=5e-5,
    batch_size=2,
    # ...
)
```

---

## Educational Value

This project demonstrates:
1. **Parameter-Efficient Fine-Tuning**: LoRA enables training large models with limited resources
2. **Domain Adaptation**: How general LLMs can be specialized for specific domains
3. **Experimental Design**: Systematic hyperparameter tuning and documentation
4. **Model Evaluation**: Multiple metrics for comprehensive assessment
5. **Practical Deployment**: From training to user-facing application

---

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size or max sequence length:
batch_size = 2
MAX_LENGTH = 300
```

### Slow Training
```python
# Reduce dataset size or epochs:
DATASET_SIZE = 1000
num_epochs = 3
```

### Model Not Improving
- Try different learning rates (1e-5 to 5e-4)
- Increase LoRA rank (r=32 or r=64)
- Train for more epochs
- Use larger dataset

### Gradio Interface Not Loading
```python
# Use different port:
demo.launch(server_port=7861)
# Or share publicly:
demo.launch(share=True)
```

---

## Author
{
  Title= Math Tutor: Fine-tuning LLM with LoRA,
  Author= KEZA PEACE,
  Year=2026,
  Howpublished= https://github.com/Peace3B/Math_Tutor_LLM_Summative_Assignment
}
```

---

## Future improvements

Areas for improvement:
- [ ] Add more math domains (linear algebra, discrete math)
- [ ] Implement answer verification system
- [ ] Add visualization for geometry problems
- [ ] Multi-turn conversation support
- [ ] Integration with educational platforms

---

## License

This project is for educational purposes. Please respect the licenses of:
- TinyLlama model
- UltraData-Math dataset
- Hugging Face libraries

---

## Acknowledgments

- **Hugging Face**: For transformers, peft, and datasets libraries
- **OpenBMB**: For the UltraData-Math dataset
- **TinyLlama Team**: For the efficient base model
- **Gradio**: For the easy-to-use interface framework

---

## Contact

For questions or issues:
- Email: p.keza@alustudent.com

---

##  Quick Commands Reference

```bash
# Install dependencies (first cell)
!pip install transformers datasets peft accelerate bitsandbytes gradio evaluate

# Load dataset
dataset = load_dataset("openbmb/UltraData-Math", split="train")

# Train model
trainer.train()

# Evaluate
metrics = evaluate_model(model, eval_dataset)

# Deploy
demo.launch(share=True)
```

---

**Happy Learning!**
