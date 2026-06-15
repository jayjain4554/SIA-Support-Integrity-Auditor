# setup_models.py
import os
if not os.path.exists('models/sia_classifier.pkl'):
    print("Models not found — running training...")
    from train_pipeline import train
    train('data/enhanced_customer_support_data.csv')