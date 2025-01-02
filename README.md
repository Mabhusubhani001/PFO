# Patient Flow Optimization 🏥

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://tensorflow.org/)


## 🏆 Tech4Tomorrow Hackathon Project
This project was developed as part of the **Tech4Tomorrow: Engineering Solutions for a Better Future** hackathon under the theme:
### "AI and Machine Learning for Healthcare"

## 🎯 Hackathon Challenge Addressed
Our team focused on revolutionizing hospital operations and patient care through advanced analytics and machine learning. We addressed critical healthcare challenges including:
- Optimizing patient flow management
- Predicting resource requirements
- Improving operational efficiency
- Enhancing decision-making through data-driven insights

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [Contact](#contact)

## 🔍 Overview

Developed during the Tech4Tomorrow hackathon, this PFO represents a comprehensive solution for healthcare facilities to optimize patient flow, predict resource requirements, and improve operational efficiency. The system combines advanced deep learning models with interactive visualizations to provide real-time insights and predictive analytics, directly addressing the challenges faced by modern healthcare institutions.

## ✨ Innovation Highlights

### 1. Healthcare AI Integration
- Deep learning models for patient flow prediction
- Resource optimization through machine learning
- Automated insight generation for decision support

### 2. Real-World Impact
- Reduction in patient wait times
- Improved resource allocation efficiency
- Enhanced operational decision-making
- Better patient care through predictive analytics

### 3. Interactive Dashboard
- Real-time monitoring of hospital metrics
- Dynamic filtering by date, ward, and weather conditions
- Responsive visualizations using Plotly
- Custom metric cards for key performance indicators

### 4. Analytics Modules
- **Patient Flow Analysis**
  - Length of stay predictions
  - Admission pattern analysis
  - Patient demographics insights
  
- **Resource Planning**
  - Bed utilization optimization
  - Staff allocation analysis
  - Equipment usage tracking
  
- **Predictive Analytics**
  - Admission forecasting
  - Resource demand prediction
  - Capacity optimization

### 5. Data Augmentation
- Seasonal pattern integration
- Weather impact analysis
- Day-of-week variation modeling

## 🧠 Model Architectures

### 1. Length of Stay Prediction Model
```python
Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
```
- Uses Huber loss for robustness against outliers
- Optimized with Adam optimizer (lr=0.001)
- Dropout layers for preventing overfitting

### 2. Admission Prediction Model
```python
Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(7, 8)),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(16)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
- Hybrid CNN-LSTM architecture for temporal pattern recognition
- Bidirectional LSTM layers for capturing complex dependencies
- Binary cross-entropy loss function

### 3. Resource Forecasting Models
- **XGBoost Regressor** for occupancy prediction
  - Parameters: max_depth=6, learning_rate=0.1, n_estimators=100
- **ARIMA Models** for time series forecasting
- **Random Forest & Gradient Boosting** for resource optimization

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hospital-analytics-dashboard.git
cd hospital-analytics-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 💻 Usage

1. Prepare your data in CSV format with required columns (see Data Requirements)
2. Configure the data path in `main()` function
3. Launch the Streamlit application
4. Use the sidebar for navigation and filtering
5. Interact with visualizations and predictions

## 📊 Data Requirements

Required columns in the input CSV:
- Admission_Date
- Discharge_Date
- Ward
- Age
- Primary_Diagnosis
- Length_of_Stay(days)
- General_Bed_Occupancy
- ICU_Bed_Occupancy
- Staff_Availability
- Weather_Conditions
- Season
- Day_of_Week

## 🔧 Technical Details

### Data Processing Pipeline
1. Data loading and validation
2. Feature engineering and augmentation
3. Scaling and normalization
   - MinMaxScaler for occupancy rates
   - StandardScaler for length of stay

### Visualization Stack
- Plotly for interactive charts
- Streamlit for UI components
- Custom CSS for styling

### Performance Optimization
- Caching for expensive computations
- Efficient data filtering
- Optimized model inference


## 🤝 Contributing

1. Star and Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📫 Contact

For technical support or queries, please reach out via:
1) Subhani ( mabhusubhani001@gmail.com )
2) Vinay ( vinaychakravarthi10110@gmail.com )

---

## 🙏 Acknowledgments
- Tech4Tomorrow Hackathon organizers
- Healthcare domain experts for valuable insights
- Open-source community for tools and libraries
- Contributors and testers

## 💡 Future Enhancements
- Integration with hospital management systems
- Mobile application development
- Enhanced predictive models
- Real-time monitoring capabilities
- Multi-hospital deployment support
