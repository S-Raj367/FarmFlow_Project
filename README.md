# FarmFlow_Project
# Intelligent Agricultural Systems: Applied Machine Learning Solutions

![Image](https://github.com/user-attachments/assets/8a410494-639a-4bc1-ab8e-9d7c963b6a90)
![Image](https://github.com/user-attachments/assets/c9cb5a68-265d-41b8-b7ef-cab80d9930e5)
![Image](https://github.com/user-attachments/assets/b18b8ce2-be44-41c5-bf8e-2be38a219050)
![Image](https://github.com/user-attachments/assets/72df65a5-21f6-42d0-920f-e0d61c800871) 
![Image](https://github.com/user-attachments/assets/6d78f640-5f71-4dc1-81fd-8136aed1debe)
![Image](https://github.com/user-attachments/assets/724d7e5b-83a8-4a0b-bd2c-ceeeb0723879)
![Image](https://github.com/user-attachments/assets/f62d4455-0bdd-409f-91cb-a2497395a4ff)
![Image](https://github.com/user-attachments/assets/54cf395c-8945-4903-857b-0357b6ef941f)
![Image](https://github.com/user-attachments/assets/ff39c0b8-5884-4245-9620-26a30be92586)





A suite of machine learning models designed to address critical challenges in modern agriculture through data-driven decision-making. This project demonstrates the practical application of AI/ML in precision agriculture, from crop recommendation to disease detection and market forecasting.

## Key Features

- **Crop Recommendation System**: Optimal crop suggestions based on environmental factors
- **Plant Disease Classifier**: Real-time disease detection from leaf images
- **Price Forecasting Engine**: Commodity market predictions
- **Growth Monitoring**: Computer vision-based plant tracking
- **Agricultural Knowledge Base**: LLM-powered farming assistant

## Technical Architecture

### Core ML Models

| Model                      | Technique                          | Performance               | Data Inputs                          |
|----------------------------|------------------------------------|---------------------------|--------------------------------------|
| Crop Recommendation        | SVM with RBF kernel               | 98.1% test accuracy       | Soil NPK, pH, weather data           |
| Plant Disease Classifier   | Fine-tuned MobileNetV2            | 95.2% validation accuracy | 54K+ leaf images (38 disease classes)|
| Price Forecasting          | Random Forest Regressor           | MAE ₹142.3                | Historical price data (2012-2019)    |
| Growth Monitoring          | OpenCV-based segmentation         | ±2.1 cm precision         | Plant images                         |
| Knowledge Base             | Llama3-70B LLM (Groq API)         | Domain-tuned responses    | Natural language queries             |

## 📊 Performance Metrics

| Model                      | Metric                          | Performance               |
|----------------------------|---------------------------------|---------------------------|
| **Crop Recommendation**    | Test Accuracy                   | 98.1%                     |
| **Disease Classification** | Validation Accuracy             | 95.2%                     |
| **Price Forecasting**      | Mean Absolute Error (MAE)       | ₹142.3 vs market data     |
| **Growth Monitoring**      | Height Estimation Error         | ±2.1 cm precision         |

## 🛠️ Technologies Used

### **Machine Learning Stack**
<p align="left">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
</p>

### **Backend Infrastructure**
- **ML Services**: Flask (Python)
- **API Layer**: Node.js/Express
- **Database**: MongoDB 
- **Authentication**: JWT Tokens


### **Frontend Ecosystem**
<p align="left">
  <img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js">
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
</p>

## 🚀 Future Enhancements

- **IoT Integration**: Real-time soil/weather sensor data pipelines
- **Satellite Analytics**: NDVI-based field health monitoring
- **Localization**: Multi-lingual support for regional farmers
- **Mobile Expansion**: React Native cross-platform app
- **Edge AI**: On-device model inference for offline use


