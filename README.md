Person-Scene Integration Web App

This is a Streamlit-based web application that allows users to upload a front-facing image of a person and a background scene. The app then extracts the person from their original image and blends them into the background in a photorealistic way using basic image processing techniques with OpenCV.

Objective

To seamlessly place a person into a given background by:
- Removing the original background of the person.
- Adjusting colors and lighting to match the background.
- Adding a basic synthetic shadow.
- Blending the person into the background using seamless cloning.

 Features

- Upload any front-facing image of a person.
- Upload a background scene image.
- Automatic background removal using Haar Cascade.
- Color harmonization between person and background.
- Shadow simulation for realistic placement.
- Seamless blending with OpenCV’s Poisson cloning.
- Real-time image generation through a simple web UI.

 Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit



Project Structure

person-scene-integration/
│
├── main.py # Streamlit app logic
├── utils.py # Utility functions for blending and masking
├── requirements.txt # All dependencies
├── haarcascade_fullbody.xml # Used for detecting person in the image
├── haarcascade_frontalface_default.xml # Optional, for facial detection
└── README.md

How to Run Locally

Clone the Repository
   
   - git clone https://github.com/kashishsingla25/person-scene-integration.git
   - cd person-scene-integration
Create a Virtual Environment (Optional)
   - python -m venv venv
   - source venv/bin/activate  # or venv\Scripts\activate on Windows

Install Dependencies
  - pip install -r requirements.txt

Run the App
  - streamlit run main.py
