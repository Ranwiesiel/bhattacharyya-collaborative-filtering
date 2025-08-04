# Collaborative Filtering with Bhattacharyya Coefficient

This repository implements an advanced collaborative filtering recommendation system based on the research paper:

**"An efficient collaborative filtering approach based on Bhattacharyya coefficient and user-item correlation"**  
*Applied Intelligence, Springer (2020)*  
DOI: [https://doi.org/10.1007/s10489-020-01775-4](https://doi.org/10.1007/s10489-020-01775-4)

## Overview
```
Note:
- This implementation does not include Congenerous Rating (CR), which is mentioned in the paper as the proposed algorithm when predicting the rating model.
- Does not include all tests for other evaluation metrics that was mentioned in the paper.
```

This implementation combines multiple advanced similarity measures to improve recommendation accuracy:

1. **Bhattacharyya Coefficient**: Measures similarity between user rating distributions
2. **Jaccard Similarity**: Captures item overlap based on user coverage  
3. **Mean-Centered Correlations**: Removes item-specific rating biases
4. **Weighted Prediction**: Uses top-k similar items for rating prediction

## Key Features

- ✅ **Advanced Similarity Measures**: Implements Bhattacharyya coefficient for user similarity
- ✅ **Mean-Centered Approach**: Removes item bias for better predictions
- ✅ **Robust Evaluation**: Hold-out validation with multiple train/test splits
- ✅ **Efficient Implementation**: Optimized NumPy operations for scalability
- ✅ **Comprehensive Documentation**: Detailed function documentation and mathematical formulas
- ✅ **Model Persistence**: Save/load trained models for production use

## Algorithm Overview

The core similarity measure combines three components:

```
Sim_BC_ADV(i,j) = Σ_u Σ_v BC(u,v) × [(r_ui - μ_i) × (r_vj - μ_j)] / (σ_i × σ_j) + Jaccard(i,j)
```

Where:
- `BC(u,v)`: Bhattacharyya coefficient between users u and v
- `r_ui`: Rating of user u for item i  
- `μ_i`: Mean rating for item i
- `σ_i`: Standard deviation of ratings for item i
- `Jaccard(i,j)`: Jaccard similarity between items i and j

## Installation

```bash
pip install numpy pandas joblib tqdm
```

## Usage

### Basic Usage

```python
import numpy as np
from implementasi import *

# Load your rating data
ratings = load("ratings.txt")  # Format: user_id item_id rating

# Compute similarity components
item_means = mean(ratings)
jc = jaccard(ratings)
bc = bhattacharyya_coefficient(ratings)

# Calculate advanced similarity matrix
sim_matrix = sim_bc_adv(ratings, item_means, jc, bc)

# Predict ratings for a user
predictions = predict_ratings_for_user(user_idx=0, ratings=ratings, 
                                     sim_matrix=sim_matrix, k=2)
```

### Model Evaluation

```python
# Perform hold-out validation
rmse = split_validation(ratings, sim_matrix, test_size=0.2, k_neighbors=2)
print(f"Model RMSE: {rmse:.4f}")
```

## Data Format

The system expects rating data in space-separated format:
```
user_id item_id rating
1 1 5.0
1 3 4.0
2 1 3.0
2 2 2.5
...
```

## Mathematical Foundation

### Bhattacharyya Coefficient
Measures similarity between probability distributions:
```
BC(u,v) = Σ √(P(r|u) × P(r|v))
```

### Mean-Centered Prediction
Removes item bias in rating prediction:
```
predicted_rating(u,i) = mean(i) + Σ(sim(i,j) × centered_rating(u,j)) / Σ|sim(i,j)|
```

### Jaccard Similarity
Captures item overlap:
```
Jaccard(i,j) = |Users(i) ∩ Users(j)| / |Users(i) ∪ Users(j)|
```

## Performance

The algorithm has been evaluated using multiple train/test splits:

- **95/5 split**: Maximum training data scenario
- **90/10 split**: Standard evaluation protocol  
- **85/15 split**: Reduced training data scenario

Lower RMSE values indicate better prediction accuracy.

## Key Advantages

1. **Distribution-Aware**: Uses Bhattacharyya coefficient to capture rating distribution patterns
2. **Bias Reduction**: Mean-centering removes item-specific rating biases
3. **Multi-Signal Fusion**: Combines multiple similarity measures for robustness
4. **Scalable**: Efficient NumPy implementation for large datasets
5. **Flexible**: Configurable neighborhood size (k-parameter)

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{bhattacharyya_cf_2020,
  title={An efficient collaborative filtering approach based on Bhattacharyya coefficient and user-item correlation},
  journal={Applied Intelligence},
  publisher={Springer},
  year={2020},
  doi={10.1007/s10489-020-01775-4},
  url={https://doi.org/10.1007/s10489-020-01775-4}
}
```

## Implementation Notes

- **Memory Efficiency**: Uses sparse matrix operations where possible
- **Numerical Stability**: Handles edge cases (zero denominators, empty distributions)
- **Validation**: Prevents data leakage in train/test splits
- **Reproducibility**: Fixed random seeds for consistent results

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for the theoretical foundation and cite appropriately.

### Academic Citation

```bibtex
   @article{bhattacharyya_cf_2020,
     title={An efficient collaborative filtering approach based on Bhattacharyya coefficient and user-item correlation},
     journal={Applied Intelligence},
     publisher={Springer},
     year={2020},
     doi={10.1007/s10489-020-01775-4}
   }
```

## Contact

For questions about this implementation or the underlying algorithm, please refer to the original research paper or create an issue in this repository.

---

**Note**: This implementation focuses on the core algorithmic contributions from the paper. For production use, consider additional optimizations for your specific use case and dataset characteristics.
