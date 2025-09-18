# 🌱 Crop Prediction Assistant

This project is a **Flask-based web app** that predicts the best crop to grow based on soil and environmental conditions.  
It also integrates with **Azure OpenAI** to provide **conversational planning assistance** — offering sowing time, irrigation, fertilizer, and pest management recommendations.

---

## ⚡ Features
- Predicts crops using a trained **RandomForestClassifier** model.
- Accepts **7 numeric input features** (like Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall).
- Provides **conversational crop planning** powered by Azure OpenAI.
- Simple and interactive **Flask web interface**.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Flask**
- **scikit-learn**
- **NumPy**
- **Azure OpenAI API**
- **HTML/CSS (Jinja2 templates)**

---

## 📂 Project Structure
├── app.py # Flask app entry point
├── model.pkl # Trained ML model for crop prediction
├── templates/
│ └── index.html # Frontend HTML template
├── .env # Environment variables (not tracked in Git)
├── .gitignore # Ignored files (includes .env, pycache)
└── README.md # Project documentation


---

## 🔑 Environment Variables

Create a **`.env`** file in the project root:

```bash
AZURE_ENDPOINT=your-azure-endpoint
AZURE_KEY=your-azure-api-key
AZURE_DEPLOYMENT=your-deployment-name
AZURE_API_VERSION=2024-02-15-preview
```

## Example Input: 21 12 34 21 34 443 1
## Output: 
The predicted crop is **rice**.
- Optimal sowing time
- Irrigation schedule
- Fertilizer recommendations
- Pest management tips

