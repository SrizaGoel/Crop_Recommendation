from flask import Flask, render_template, request
import joblib
import numpy as np
import requests
import os
from flask import make_response, render_template_string
from xhtml2pdf import pisa
from io import BytesIO


PIXELS_API_KEY = "OtEtZrY41XcPLOo7SAktHs51GEY3oYDvDlksyUPbJNNoihP34xhCAWaj"
fallback_images = {
    "orange": "https://cdn.britannica.com/24/174524-050-A851D3F2/Oranges.jpg",
    "kidneybeans": "https://media.istockphoto.com/id/541973406/photo/red-bean-on-wooden-bowl-with-spoon.jpg?s=612x612&w=0&k=20&c=bsdH0SA1hfpSfO2FEdUmE9sD0_p-Dq0dEVvvWyS7W1A=",
    "pigeonpeas": "https://media.istockphoto.com/id/1163019269/photo/organic-dry-toor-whole-pigeon-peas.jpg?s=612x612&w=0&k=20&c=ingKrShpFvdZwC4Lj6oNTwHU0BL7zyraPaQMTurlsbY=",
    "mothbeans": "https://www.aranyapurefood.com/cdn/shop/files/WhatsAppImage2024-03-27at10.19.02AM.jpg?v=1711515146",
    "mungbean": "https://media.post.rvohealth.io/wp-content/uploads/sites/3/2020/02/324156_2200-800x1200.jpg",
    "blackgram": "https://www.netmeds.com/images/cms/wysiwyg/blog/2019/09/BlackGram_big_1.jpg",
    "lentil": "https://millychino.com/wp-content/uploads/what-are-lentils-0.jpg",
    "banana": "https://www.koppert.in/content/_processed_/3/9/csm_Banana_8775c22bb9.jpg"
}

app = Flask(__name__)
model = joblib.load('rf_pipeline.pkl')
def fetch_crop_image(crop_name):
    # Always use fallback for certain crops
    if crop_name.lower() in fallback_images:
        return fallback_images[crop_name.lower()]

    url = f"https://api.pexels.com/v1/search?query={crop_name}&per_page=1"
    headers = {
        "Authorization": PIXELS_API_KEY
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if "photos" in data and data["photos"]:
            return data["photos"][0]["src"]["landscape"]
        else:
            return fallback_images.get(crop_name.lower(), "")
    except Exception as e:
        print("Image fetch error:", e)
        return fallback_images.get(crop_name.lower(), "")




@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:

        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        # Get top 3 predictions sorted by probability
        top_indices = np.argsort(probabilities)[::-1][:4]
        top_preds = [(classes[i], round(probabilities[i]*100, 2)) for i in top_indices]

        # Top prediction
        top_crop, top_conf = top_preds[0]
        prediction_text = f"Top Recommended Crop: {top_crop} ({top_conf}% confidence)"
        image_url = fetch_crop_image(top_crop)

        # Exclude top crop from list of 3
        other_preds = top_preds[1:]  # show only 2 alternatives
        print("Top crop:", top_crop)
        print("Image URL:", image_url)

        return render_template('index.html',
                               prediction_text=prediction_text,
                               crop_image_url=image_url,
                               top_predictions=other_preds,
                               request=request,
                               top_crop=top_crop,         
                               top_conf=top_conf)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/download", methods=["POST"])
def download():
    try:
        crop = request.form.get("crop")
        confidence = request.form.get("confidence")
        inputs = {
            "N": request.form.get("N"),
            "P": request.form.get("P"),
            "K": request.form.get("K"),
            "Temperature": request.form.get("temperature"),
            "Humidity": request.form.get("humidity"),
            "pH": request.form.get("ph"),
            "Rainfall": request.form.get("rainfall"),
            "Location": request.form.get("location")
        }

        template = """
        <h2>🌾 Crop Recommendation Report</h2>
        <p><b>Predicted Crop:</b> {{ crop }} ({{ confidence }}% confidence)</p>
        <h3>Input Summary:</h3>
        <ul>
        {% for key, value in inputs.items() %}
          <li><b>{{ key }}:</b> {{ value }}</li>
        {% endfor %}
        </ul>
        """

        rendered = render_template_string(template, crop=crop, confidence=confidence, inputs=inputs)
        pdf = BytesIO()
        pisa.CreatePDF(rendered, dest=pdf)

        response = make_response(pdf.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=crop_report.pdf"
        return response
    except Exception as e:
        return f"Error generating PDF: {e}"



if __name__ == "__main__":
    app.run(debug=True)
