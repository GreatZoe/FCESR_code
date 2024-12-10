**The complete code will be released upon acceptance**

1. explain_model.py: Trains the model and generates explanations.
2. contrastive_fine_tune.py: Utilizes the generated explanations to enhance the recommendation performance.
3. train_model.py: Trains the model and saves the trained model.

**Dependencies**

- **Python version**: 3.8.19
- **Main dependencies**:
  - `torch==1.8.0`
  - `numpy==1.24.3`

**Example:**
* baby/narm directory contains the results of training using the NARM model.
  * recommend_model.ckpt: The model before fine-tuning.
  * best_model.ckpt: The model after fine-tuning.