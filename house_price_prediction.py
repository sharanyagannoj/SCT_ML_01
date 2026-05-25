"""
Linear Regression Model for House Price Prediction
Predicts house prices based on square footage, bedrooms, and bathrooms.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def create_sample_dataset():
    """Create a sample house price dataset."""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 100
    square_footage = np.random.uniform(800, 5000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples)
    
    # Generate prices based on features (with some noise)
    # Price = 50 * sqft + 20000 * bedrooms + 15000 * bathrooms + noise
    price = (50 * square_footage + 20000 * bedrooms + 15000 * bathrooms + 
             np.random.normal(0, 50000, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'price': price
    })
    
    return data


def train_and_evaluate_model(X, y):
    """Train the linear regression model and evaluate it."""
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }


def display_model_info(model, metrics):
    """Display model coefficients and performance metrics."""
    print("=" * 60)
    print("LINEAR REGRESSION MODEL - HOUSE PRICE PREDICTION")
    print("=" * 60)
    print("\nModel Coefficients:")
    print(f"  Square Footage:  ${model.coef_[0]:.2f} per sqft")
    print(f"  Bedrooms:        ${model.coef_[1]:.2f} per bedroom")
    print(f"  Bathrooms:       ${model.coef_[2]:.2f} per bathroom")
    print(f"  Intercept:       ${model.intercept_:.2f}")
    
    print("\n" + "-" * 60)
    print("Model Performance Metrics:")
    print("-" * 60)
    print(f"Training Set:")
    print(f"  RMSE: ${metrics['train_rmse']:.2f}")
    print(f"  MAE:  ${metrics['train_mae']:.2f}")
    print(f"  R²:   {metrics['train_r2']:.4f}")
    
    print(f"\nTesting Set:")
    print(f"  RMSE: ${metrics['test_rmse']:.2f}")
    print(f"  MAE:  ${metrics['test_mae']:.2f}")
    print(f"  R²:   {metrics['test_r2']:.4f}")
    print("=" * 60)


def make_predictions(model, new_data):
    """Make predictions for new house data."""
    print("\nPredicting Prices for New Houses:")
    print("-" * 60)
    
    predictions = model.predict(new_data)
    
    for i, (idx, row) in enumerate(new_data.iterrows()):
        print(f"House {i + 1}:")
        print(f"  Square Footage: {row['square_footage']:.0f} sqft")
        print(f"  Bedrooms:       {row['bedrooms']:.0f}")
        print(f"  Bathrooms:      {row['bathrooms']:.1f}")
        print(f"  Predicted Price: ${predictions[i]:,.2f}")
        print()


def plot_results(y_test, y_pred_test):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'price_predictions.png'")
    plt.close()


def main():
    """Main function to run the entire pipeline."""
    # Step 1: Create or load data
    print("Loading dataset...")
    data = create_sample_dataset()
    print(f"Dataset created with {len(data)} samples\n")
    
    # Step 2: Prepare features and target
    X = data[['square_footage', 'bedrooms', 'bathrooms']]
    y = data['price']
    
    # Step 3: Train and evaluate model
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, metrics = \
        train_and_evaluate_model(X, y)
    
    # Step 4: Display model information
    display_model_info(model, metrics)
    
    # Step 5: Make predictions on new data
    new_houses = pd.DataFrame({
        'square_footage': [2000, 3500, 1500],
        'bedrooms': [3, 4, 2],
        'bathrooms': [2.0, 3.5, 1.5]
    })
    make_predictions(model, new_houses)
    
    # Step 6: Visualize results
    plot_results(y_test, y_pred_test)
    
    # Step 7: Save the dataset for reference
    data.to_csv('house_price_dataset.csv', index=False)
    print("Dataset saved as 'house_price_dataset.csv'")


if __name__ == "__main__":
    main()
