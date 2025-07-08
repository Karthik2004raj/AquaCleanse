# AquaCleanse

AquaCleanse is a smart water and waste management platform that leverages machine learning and web technologies to classify waste, analyze water quality, and provide actionable insights for environmental health. This project is designed for both hardware-based and web-based applications, supporting multi-label image classification and water quality prediction.

## Features
- **Multi-label Waste Classification:** Uses a Convolutional Neural Network (CNN) with transfer learning (ResNet50) to classify images of waste into categories such as plastic, metal, biomedical, and shoes.
- **Water Quality Analysis:** Analyzes water quality data (BOD, DO, etc.) and predicts potential health risks and solutions.
- **User and Admin Authentication:** Secure login and signup for users and admins.
- **Email Notifications:** Sends price quotes and notifications via email.
- **Interactive Web Dashboard:** Built with Flask, providing user-friendly interfaces for uploading images, viewing reports, and managing data.

## Project Structure
```
AquaCleanse/
  app/
    __init__.py
    train_cnn.py
    routes.py
    running.py
    static/
    templates/
  ...
README.md
```

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/Karthik2004raj/AquaCleanse.git
cd AquaCleanse
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare the Dataset
- Place your image dataset in the appropriate directory (update `DATASET_DIR` in `train_cnn.py`).
- Ensure `static/water_quality.csv` exists for water quality analysis.

### 4. Train the Model (Optional)
To train the CNN model for waste classification:
```sh
python app/train_cnn.py
```
This will save the trained model as `waste_cnn_model.h5`.

### 5. Run the Web Application
```sh
python app/running.py
```
The app will be available at `http://localhost:5000`.

## Usage
- **User:** Sign up or log in to upload waste images, view water quality reports, and receive recommendations.
- **Admin:** Log in to manage users, view analytics, and oversee system operations.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch for your feature or fix, and submit a pull request.

## License
This project is licensed under the MIT License.
