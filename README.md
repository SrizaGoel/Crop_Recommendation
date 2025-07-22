# Smart Crop Recommender

An ML-based web application to suggest the most suitable crop for your land based on soil and climate parameters like N, P, K, temperature, humidity, pH, and rainfall.

## Live Demo
👉 [Click to View](https://crop-recommendation-jg7l.onrender.com)

## Features
- Predicts the best crop based on input features
- Shows top 3 crop suggestions with confidence
- Option to download PDF report

## Input Features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)

## Tech Stack
- **Frontend**: HTML5, CSS3
- **Backend**: Python, Flask
- **ML Model**: RandomForestClassifier (scikit-learn)
- **Deployment**: Render

## Project Structure
```bash
crop-recommendation/
│
├── templates/                  
│   └── index.html              
│
├── model/                      
│   └── rf_pipeline.pkl               
│
├── app.py                     
├── requirements.txt            
├── README.md                                   
```


## ⚙️ Setup Instructions

```bash
git clone https://github.com/SrizaGoel/Crop_Recommendation.git
cd Crop_Recommendation
pip install -r requirements.txt
python app.py

