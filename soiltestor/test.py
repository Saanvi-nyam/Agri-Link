import pickle
import json

# Load your model
with open(r'C:\Users\Lab\Documents\GitHub\Agri-link\soiltestor\NPKModel.pkl', 'rb') as f:
    model = pickle.load(f)

# Example for a scikit-learn model
model_params = {
    'coef_': model.coef_.tolist(),  # For linear models
    'intercept_': model.intercept_.tolist(),  # For linear models
    # Add other necessary parameters specific to your model
}

# Save to JSON
with open('model.json', 'w') as json_file:
    json.dump(model_params, json_file)
