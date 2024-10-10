# Classification Model for Imbalanced Dataset

I wanted to try to building a classification model for an imbalanced dataset.
The dataset I am working with is a large dataset of amazon products with these features
  product_id - Product ID
  product_name - Name of the Product
  category - Category of the Product
  discounted_price - Discounted Price of the Product
  actual_price - Actual Price of the Product
  discount_percentage - Percentage of Discount for the Product
  rating - Rating of the Product
  rating_count - Number of people who voted for the Amazon rating
  about_product - Description about the Product
  user_id - ID of the user who wrote review for the Product
  user_name - Name of the user who wrote review for the Product
  review_id - ID of the user review
  review_title - Short review
  review_content - Long review
  img_link - Image Link of the Product
  product_link - Official Website Link of the Product

The model is trying to classify Amazon products based on the number of ratings they have received into one of two categories or "bins." Specifically, it categorizes each product’s `rating_count` into:

- **Bin1 (Low Rating Count)**: Products with a low number of ratings.
- **Bin2 (High Rating Count)**: Products with a high number of ratings.

The goal of the model is to predict which bin a product falls into based on features such as the product’s discounted price, actual price, discount percentage, category, overall rating, and sentiment of reviews.

## Steps Taken, Challenges Encountered, and Methods Used

### 1. Initial Model Training and Class Imbalance Issue
- The dataset was imbalanced for this purpose, with a majority class (Bin1) having 264 samples and a minority class (Bin2) having only 29 samples. This caused most models, including Logistic Regression, SVC, Gradient Boosting, and others, to have high accuracy (close to 90%), but they failed to predict the minority class (Bin2) effectively.
- The models were biased toward predicting the majority class due to this imbalance, making the high accuracy misleading.

### 2. Evaluation Metrics and Observations
- Some models, such as SVC and Gradient Boosting, had zero precision and recall for the minority class, indicating that these models were not making any correct predictions for Bin2.
- Logistic Regression and KNeighborsClassifier models also showed low recall or low precision for the minority class, highlighting a recurring issue with false positives or missed predictions for Bin2.
- Gaussian Naive Bayes behaved unusually, predicting the minority class more often but with a significant drop in accuracy (around 44%). This was due to its assumptions of feature independence and a Gaussian distribution, which may not fit the dataset.

### 3. Applying SMOTE to Balance Classes
- To address the imbalance, I used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class until it matched the size of the majority class, making the dataset balanced with 1038 samples per class.
- After applying SMOTE, I noticed a drop in accuracy for some models, like Logistic Regression (to around 64.5%), indicating that the model was now more sensitive to the minority class but struggled with precision, resulting in many false positives.

### 4. Performance Analysis After SMOTE
- **Logistic Regression**: The model’s recall for the minority class improved (0.59), but precision remained low (0.16). The accuracy drop suggested that the model was less biased but sacrificed overall correctness to identify minority class instances.
- **Random Forest**: The model maintained a high accuracy (90.4%) but still had low recall for the minority class (0.14). This indicated that the model was still biased towards predicting the majority class despite the SMOTE adjustment.
- **Support Vector Classifier (SVC)**: The accuracy dropped to 68.6%. It achieved a higher recall for the minority class (0.59) compared to other models, but precision remained low (0.18). This trade-off showed that while the model identified more minority class instances, it also misclassified many of them.

### 5. Using Class Weights and Stratified Cross-Validation
- To further address class imbalance, I trained models using class weights (e.g., `class_weight='balanced'`), which allowed the models to give more importance to the minority class.
- Stratified cross-validation was applied to maintain consistent class proportions across training and validation sets. This provided a more realistic evaluation of the models, ensuring that the minority class was always represented in the splits.

### 6. Further Evaluations with Alternative Metrics
- In addition to accuracy, I evaluated the models using metrics like ROC AUC, Precision-Recall AUC, and F1 Score, as they provide a better understanding of model performance when dealing with imbalanced datasets.
- For example:
  - The ROC AUC score of 0.71 indicated moderate performance in distinguishing between classes.
  - The precision-recall AUC was quite low (0.20), showing that the models still struggled to balance precision and recall.
  - The F1 Score (0.59) highlighted that the models’ performance was imbalanced between classes.

### 7. Summary of Challenges and Observations
- **Class Imbalance**: Despite using SMOTE, models like Random Forest maintained high accuracy while still failing to identify minority class instances correctly.
- **Low Precision**: The low precision for the minority class persisted across models, indicating a common issue with false positives even after resampling.
- **Varying Model Behaviors**:
  - Logistic Regression showed better recall balance but had lower precision.
  - SVC had some success with minority class recall but compromised overall accuracy.
  - Random Forest performed reliably in cross-validation but remained biased toward the majority class.

### 8. Next Steps for Improvement
- I plan to further tune models using techniques like grid search or random search to find the optimal hyperparameters for Random Forest, SVC, and Logistic Regression.
- Exploring ensemble methods like `BalancedRandomForestClassifier` or `EasyEnsemble` could help improve minority class prediction.
- Adjusting SMOTE parameters with variants such as SMOTE-Tomek or SMOTE-ENN could refine how synthetic data is generated.
- Additional focus on metrics like Precision-Recall AUC, F1-Score for the minority class, and confusion matrix analysis was recommended over accuracy alone to get a more complete picture of model performance.

### Conclusion
By following these steps and addressing the challenges, I learned how to handle class imbalances, use advanced metrics for evaluation, and iterate on models to improve their ability to predict minority classes more accurately. This hands-on approach deepened my understanding of machine learning, especially with imbalanced datasets.
