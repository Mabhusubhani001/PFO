import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DataAugmenter:
    """Enhanced data augmentation specific to hospital data"""
    
    @staticmethod
    def add_seasonal_patterns(df):
        """Add realistic seasonal variations"""
        season_factors = {
            'Winter': {
                'General_Bed_Occupancy': 1.3,  # Higher in winter
                'ICU_Bed_Occupancy': 1.4,      # Even higher ICU demand
                'Staff_Availability': 0.9       # Lower due to holidays
            },
            'Summer': {
                'General_Bed_Occupancy': 0.8,   # Lower in summer
                'ICU_Bed_Occupancy': 0.9,
                'Staff_Availability': 1.1       # Better availability
            },
            'Spring': {
                'General_Bed_Occupancy': 1.0,
                'ICU_Bed_Occupancy': 1.0,
                'Staff_Availability': 1.0
            },
            'Fall': {
                'General_Bed_Occupancy': 1.1,
                'ICU_Bed_Occupancy': 1.1,
                'Staff_Availability': 0.95
            }
        }
        
        for season, factors in season_factors.items():
            mask = df['Season'] == season
            for column, factor in factors.items():
                noise = np.random.normal(0, 0.05, size=len(df[mask]))
                df.loc[mask, column] *= (factor + noise)
        
        return df
    
    @staticmethod
    def add_weather_patterns(df):
        """Add realistic weather-based variations"""
        weather_impacts = {
            'Rainy': {
                'General_Bed_Occupancy': 1.1,
                'ICU_Bed_Occupancy': 1.2,
                'Length_of_Stay(days)': 1.05
            },
            'Stormy': {
                'General_Bed_Occupancy': 1.3,
                'ICU_Bed_Occupancy': 1.4,
                'Length_of_Stay(days)': 1.1
            },
            'Clear': {
                'General_Bed_Occupancy': 0.9,
                'ICU_Bed_Occupancy': 0.95,
                'Length_of_Stay(days)': 0.95
            }
        }
        
        for weather, impacts in weather_impacts.items():
            mask = df['Weather_Conditions'] == weather
            for metric, factor in impacts.items():
                noise = np.random.normal(0, 0.03, size=len(df[mask]))
                df.loc[mask, metric] *= (factor + noise)
        
        return df
    
    @staticmethod
    def add_day_of_week_patterns(df):
        """Add realistic day of week patterns"""
        dow_patterns = {
            'Monday': 1.2,    # Busiest day
            'Tuesday': 1.1,
            'Wednesday': 1.0,
            'Thursday': 0.95,
            'Friday': 0.9,
            'Saturday': 0.7,  # Quieter weekends
            'Sunday': 0.75
        }
        
        for day, factor in dow_patterns.items():
            mask = df['Day_of_Week'] == day
            noise = np.random.normal(0, 0.05, size=len(df[mask]))
            df.loc[mask, 'General_Bed_Occupancy'] *= (factor + noise)
            df.loc[mask, 'Staff_Availability'] *= (factor + noise)
        
        return df

class HospitalAnalytics:
    """Advanced analytics for hospital data"""
    
    def __init__(self):
        self.scalers = {
            'occupancy': MinMaxScaler(),
            'los': StandardScaler()
        }
        self.encoders = {}
        self.models = self._initialize_models()
    
    def _initialize_models(self):
        """Initialize prediction models"""
        return {
            'los': self._build_los_model(),
            'occupancy': self._build_occupancy_model(),
            'admission': self._build_admission_model()
        }
    
    def _build_los_model(self):
        """Build Length of Stay prediction model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='huber')
        return model
    
    def _build_occupancy_model(self):
        """Build occupancy prediction model"""
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    
    def _build_admission_model(self):
        """Build admission prediction model"""
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(7, 8)),
            MaxPooling1D(pool_size=2),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(16)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
        return model

class HospitalDashboard:
    """Enhanced hospital analytics dashboard"""
    
    def __init__(self, data_path):
        self.setup_streamlit()
        self.load_data(data_path)
        self.analytics = HospitalAnalytics()
        self.filtered_data = None
    
    def setup_streamlit(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Hospital Analytics Dashboard",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom styling
        st.markdown("""
            <style>
                .reportview-container {
                    background: linear-gradient(to right, #f8f9fa, #ffffff);
                }
                .metric-container {
                    background-color: black;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .insight-box {
                    background-color: black;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #0066cc;
                    margin: 10px 0;
                }
            </style>
        """, unsafe_allow_html=True)
    
    def load_data(self, data_path):
        """Load and prepare data"""
        df = pd.read_csv(data_path)
        
        # Convert dates
        df['Admission_Date'] = pd.to_datetime(df['Admission_Date'], dayfirst=True, errors='coerce')
        df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'], dayfirst=True, errors='coerce')
        
        # Add synthetic patterns
        augmenter = DataAugmenter()
        df = augmenter.add_seasonal_patterns(df)
        df = augmenter.add_weather_patterns(df)
        df = augmenter.add_day_of_week_patterns(df)
        
        # Calculate derived metrics
        df['Occupancy_Rate'] = ((df['General_Bed_Occupancy'] + df['ICU_Bed_Occupancy']) / 
                               df['Daily_Bed_Availability'])
        df['Staff_Per_Patient'] = df['Staff_Availability'] / (df['General_Bed_Occupancy'] + 
                                                            df['ICU_Bed_Occupancy'])
        df['Total_Occupancy'] = df['General_Bed_Occupancy'] + df['ICU_Bed_Occupancy']
        
        self.data = df
        self.filtered_data = df.copy()
    
    def run(self):
        """Run the dashboard"""
        with st.sidebar:
            self.render_sidebar()
        
        # Get selected page
        page = st.session_state.get('page', 'Overview')
        
        # Apply filters only for specific pages
        if page in ['Overview', 'Patient Analytics', 'Resource Planning']:
            self.filtered_data = self.apply_filters()
        else:
            self.filtered_data = self.data.copy()
        
        # Render selected page
        if page == 'Overview':
            self.overview_page()
        elif page == 'Patient Analytics':
            self.patient_analytics_page()
        elif page == 'Resource Planning':
            self.resource_planning_page()
        elif page == 'Predictions':
            self.predictions_page()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.title("Patient Flow Optimization")
        
        # Navigation
        st.session_state.page = st.radio(
            " ",
            ['Overview', 'Patient Analytics', 'Resource Planning', 'Predictions']
        )
        
        # Only show filters for specific pages
        if st.session_state.page in ['Overview', 'Patient Analytics', 'Resource Planning']:
            st.subheader("Filters")
            
            # Store filter values in session state
            if 'date_range' not in st.session_state:
                st.session_state.date_range = [
                    self.data['Admission_Date'].min(),
                    self.data['Admission_Date'].max()
                ]
            
            st.session_state.date_range = st.date_input(
                "Date Range",
                st.session_state.date_range
            )
            
            st.session_state.selected_wards = st.multiselect(
                "Select Wards",
                options=self.data['Ward'].unique()
            )
            
            st.session_state.selected_weather = st.multiselect(
                "Weather Conditions",
                options=self.data['Weather_Conditions'].unique()
            )
    
    def apply_filters(self):
        """Apply selected filters to data"""
        filtered_data = self.data.copy()
        
        if len(st.session_state.date_range) == 2:
            filtered_data = filtered_data[
                (filtered_data['Admission_Date'].dt.date >= st.session_state.date_range[0]) &
                (filtered_data['Admission_Date'].dt.date <= st.session_state.date_range[1])
            ]
        
        if st.session_state.selected_wards:
            filtered_data = filtered_data[filtered_data['Ward'].isin(st.session_state.selected_wards)]
        
        if st.session_state.selected_weather:
            filtered_data = filtered_data[filtered_data['Weather_Conditions'].isin(st.session_state.selected_weather)]
        
        return filtered_data
    
    def overview_page(self):
        """Render overview page"""
        st.title("Hospital Operations Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.metric_card(
                "Current Occupancy",
                f"{self.filtered_data['Occupancy_Rate'].iloc[-1]*100:.1f}%",
                "Target: 85%"
            )
        
        with col2:
            self.metric_card(
                "Avg Length of Stay",
                f"{self.filtered_data['Length_of_Stay(days)'].mean():.1f} days",
                "vs 4.5 target"
            )
        
        with col3:
            self.metric_card(
                "Staff per Patient",
                f"{self.filtered_data['Staff_Per_Patient'].mean():.2f}",
                "Target: 2.0"
            )
        
        with col4:
            self.metric_card(
                "ICU Utilization",
                f"{(self.filtered_data['ICU_Bed_Occupancy'].sum() / self.filtered_data['Daily_Bed_Availability'].sum()*100):.1f}%",
                "Target: 75%"
            )
        
        # Trends
        self.render_trends()
        
        # Insights
        self.render_insights()
    # [Previous code remains exactly the same until the overview_page method]

    def patient_analytics_page(self):
        """Render patient analytics page"""
        st.title("Patient Analytics")
        
        # Age Distribution
        st.subheader("Age Distribution Analysis")
        st.write("This histogram shows the distribution of patient ages, highlighting demographic patterns and age-related trends in the dataset.")

        fig = px.histogram(
            self.filtered_data,
            x='Age',
            nbins=30,
            title="Patient Age Distribution",
            color_discrete_sequence=['#3366cc']
        )
        fig.update_layout(
            xaxis_title="Age",
            yaxis_title="Number of Patients",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Length of Stay Analysis
        st.subheader("Length of Stay Analysis")
        st.write("""
This section examines the length of hospital stays, segmented by ward and season. The box plots provide insights into the variation and distribution of stay durations in different wards and across seasons.
""")
        col1, col2 = st.columns(2)
        
        with col1:
            # LOS by Ward
            fig = px.box(
                self.filtered_data,
                x='Ward',
                y='Length_of_Stay(days)',
                title="Length of Stay by Ward"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # LOS by Season
            fig = px.box(
                self.filtered_data,
                x='Season',
                y='Length_of_Stay(days)',
                title="Length of Stay by Season"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Primary Diagnosis Distribution
#         st.subheader("Primary Diagnosis Analysis")
#         st.write("""
# This bar chart displays the top 10 primary diagnoses, helping identify the most common health conditions among patients.
# """)
#         diagnosis_counts = self.filtered_data['Primary_Diagnosis'].value_counts().head(10)
#         fig = px.bar(
#             diagnosis_counts,
#             title="Top 10 Primary Diagnoses",
#             labels={'index': 'Diagnosis', 'value': 'Count'}
#         )
#         st.plotly_chart(fig, use_container_width=True)
        st.subheader("Primary Diagnosis Analysis")
        st.write("""
        This bar chart displays the top 10 primary diagnoses, helping identify the most common health conditions among patients.
        """)

        # Count the occurrences of each diagnosis and take the top 10
        diagnosis_counts = self.filtered_data['Primary_Diagnosis'].value_counts().head(10)

        # Add light random noise to the counts
        noise = np.random.uniform(-0.1, 0.1, size=len(diagnosis_counts))  # Add noise between -10% and +10%
        adjusted_counts = diagnosis_counts * (1 + noise)
        adjusted_counts = adjusted_counts.round()  # Round the counts to make them integers

        # Create the bar chart with adjusted counts
        fig = px.bar(
            adjusted_counts,
            title="Top 10 Primary Diagnoses",
            labels={'index': 'Diagnosis', 'value': 'Count'}
        )

        st.plotly_chart(fig, use_container_width=True)
    
    def resource_planning_page(self):
        """Render resource planning page"""
        st.title("Resource Planning & Optimization")
        
        # Bed Utilization
        st.subheader("Bed Utilization Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # General vs ICU Bed Occupancy
            bed_data = pd.DataFrame({
                'Bed Type': ['General', 'ICU'],
                'Average Occupancy': [
                    self.filtered_data['General_Bed_Occupancy'].mean(),
                    self.filtered_data['ICU_Bed_Occupancy'].mean()
                ]
            })
            fig = px.bar(
                bed_data,
                x='Bed Type',
                y='Average Occupancy',
                title="Average Bed Occupancy by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Occupancy Trends
            fig = px.line(
                self.filtered_data,
                x='Admission_Date',
                y=['General_Bed_Occupancy', 'ICU_Bed_Occupancy'],
                title="Bed Occupancy Trends"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Staff Analysis
        st.subheader("Staff Allocation Analysis")
        
        # Staff availability by day of week
        staff_by_day = self.filtered_data.groupby('Day_of_Week')['Staff_Availability'].mean()
        fig = px.bar(
            staff_by_day,
            title="Average Staff Availability by Day",
            labels={'value': 'Average Staff Available'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Staff to patient ratio analysis
        

        st.subheader("Staff-to-Patient Ratio Analysis")
        fig = px.scatter(
            self.filtered_data,
            x='Total_Occupancy',
            y='Staff_Availability',
            color='Ward',
            title="Staff vs Occupancy by Ward"
        )
        st.plotly_chart(fig, use_container_width=True)
        

    def predictions_page(self):
        """Enhanced interactive predictions and optimization page"""
        st.title("Predictive Analytics & Flow Optimization")
        st.write("""
    This section provides advanced predictive analytics for hospital management, focusing on predicting admissions, forecasting resource demands, and planning capacity optimization. 
    By inputting specific parameters, users can gain insights into patient admissions, resource availability, and optimal capacity management.
    """)
        st.sidebar.header("Contact Information")
        st.sidebar.info("""
        For more information or technical support:
        * Email 1: mabhusubhani001@gmail.com 
        * Email 2: vinaychakravarthi10110@gmail.com
        """)
        
        # Create tabs for different prediction features
        pred_tab1, pred_tab2, pred_tab3,pred_tab4 = st.tabs([
            "Admission Prediction", 
            "Resource Forecasting",
            "Capacity Planning",
            "Admission Forecasting"
        ])
        with pred_tab1:
            self._admission_prediction_section()
        
        with pred_tab2:
            self._resource_forecasting_section()
            
        with pred_tab3:
            self._capacity_planning_section()
        with pred_tab4:
            self._admission_forecasting_section()

    # def _admission_prediction_section(self):
    #     """Interactive admission prediction section"""
    #     st.subheader("Admission Prediction & Planning")
    #     st.write("""
    # Predict the required resources for patient admissions based on key parameters like age, diagnosis, ward, and season. 
    # Explore how various factors impact the length of stay, occupancy rates, and optimal admission days.
    # """)
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         # Input parameters
    #         age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
    #         diagnosis = st.selectbox(
    #             "Primary Diagnosis",
    #             sorted(self.filtered_data['Primary_Diagnosis'].unique())
    #         )
    #         ward = st.selectbox(
    #             "Preferred Ward",
    #             sorted(self.filtered_data['Ward'].unique())
    #         )
            
    #     with col2:
    #         season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    #         urgency = st.selectbox(
    #             "Admission Type",
    #             ["Emergency", "Urgent", "Regular", "Planned"]
    #         )
    #         special_requirements = st.multiselect(
    #             "Special Requirements",
    #             ["ICU Care", "Special Equipment", "Isolation", "None"]
    #         )

    #     if st.button("Predict Admission Requirements"):
    #         # Calculate predictions
    #         prediction_results = self._calculate_admission_predictions(
    #             age, diagnosis, ward, season, urgency, special_requirements
    #         )
            
    #         # Display prediction results
    #         self._display_admission_predictions(prediction_results)

    # def _calculate_admission_predictions(self, age, diagnosis, ward, season, urgency, special_requirements):
    #     """Calculate admission predictions based on inputs"""
    #     # Filter relevant historical data
    #     relevant_data = self.filtered_data[
    #         (self.filtered_data['Ward'] == ward) &
    #         (self.filtered_data['Season'] == season)
    #     ]
        
    #     # Calculate predicted metrics
    #     predicted_los = relevant_data['Length_of_Stay(days)'].mean()
    #     predicted_occupancy = relevant_data['Occupancy_Rate'].mean()
        
    #     # Adjust based on urgency
    #     urgency_factors = {
    #         "Emergency": 1.2,
    #         "Urgent": 1.1,
    #         "Regular": 1.0,
    #         "Planned": 0.9
    #     }
        
    #     predicted_los *= urgency_factors[urgency]
        
    #     # Calculate optimal admission time
    #     day_occupancy = relevant_data.groupby('Day_of_Week')['Occupancy_Rate'].mean()
    #     optimal_day = day_occupancy.idxmin()
        
    #     return {
    #         'predicted_los': predicted_los,
    #         'predicted_occupancy': predicted_occupancy,
    #         'optimal_day': optimal_day,
    #         'resource_availability': self._calculate_resource_availability(ward, special_requirements)
    #     }
    def _admission_prediction_section(self):
        """Interactive admission prediction section"""
        st.subheader("Admission Prediction & Planning")
    
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
            diagnosis = st.selectbox(
                "Primary Diagnosis",
                sorted(self.filtered_data['Primary_Diagnosis'].unique())
            )
            ward = st.selectbox(
                "Preferred Ward",
                sorted(self.filtered_data['Ward'].unique())
            )
            
        with col2:
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
            urgency = st.selectbox(
                "Admission Type",
                ["Emergency", "Urgent", "Regular", "Planned"]
            )
            special_requirements = st.multiselect(
                "Special Requirements",
                ["ICU Care", "Special Equipment", "Isolation", "None"]
            )
            
            # Month selector
            month = st.selectbox(
                "Select Month for Prediction",
                ["May 2017", "June 2017", "July 2017", "August 2017", "September 2017", 
                "October 2017", "November 2017", "December 2017"]
            )
            
        if st.button("Predict Admission Requirements"):
            # Calculate predictions
            prediction_results = self._calculate_admission_predictions(
                age, diagnosis, ward, season, urgency, special_requirements, month
            )
            
            # Display prediction results
            self._display_admission_predictions(prediction_results)

    def _calculate_admission_predictions(self, age, diagnosis, ward, season, urgency, special_requirements, month):
        """Calculate admission predictions based on inputs and month"""
        # Filter relevant historical data
        relevant_data = self.filtered_data[
            (self.filtered_data['Ward'] == ward) &
            (self.filtered_data['Season'] == season)
        ]
        
        # Add logic to handle months after May 2017
        if month and month > "May 2017":
            st.warning(f"Predictions beyond May 2017 are based on extrapolated data.")
        
        # Calculate predicted metrics (same logic as before)
        predicted_los = relevant_data['Length_of_Stay(days)'].mean()
        predicted_occupancy = relevant_data['Occupancy_Rate'].mean()
        
        # Adjust based on urgency
        urgency_factors = {
            "Emergency": 1.2,
            "Urgent": 1.1,
            "Regular": 1.0,
            "Planned": 0.9
        }
        
        predicted_los *= urgency_factors[urgency]
        
        # Calculate optimal admission time (same logic as before)
        day_occupancy = relevant_data.groupby('Day_of_Week')['Occupancy_Rate'].mean()
        optimal_day = day_occupancy.idxmin()
        
        return {
            'predicted_los': predicted_los,
            'predicted_occupancy': predicted_occupancy,
            'optimal_day': optimal_day,
            'resource_availability': self._calculate_resource_availability(ward, special_requirements)
        }

    
    # def _display_admission_predictions(self, results):
    #     """Display admission prediction results with interactive visualizations"""
    #     st.markdown("### Prediction Results")
        
    #     # Display key metrics in columns
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.metric(
    #             "Predicted Length of Stay",
    #             f"{results['predicted_los']:.1f} days",
    #             f"{results['predicted_los'] - 4.5:.1f} vs target"
    #         )
            
    #     with col2:
    #         st.metric(
    #             "Expected Occupancy",
    #             f"{results['predicted_occupancy']:.1%}",
    #             f"{(results['predicted_occupancy'] - 0.85)*100:.1f}% vs target"
    #         )
            
    #     with col3:
    #         st.metric(
    #             "Optimal Admission Day",
    #             results['optimal_day'],
    #             "Recommended"
    #         )
        
    #     # Resource availability chart
    #     st.subheader("Resource Availability Forecast")
    #     fig = px.bar(
    #         pd.DataFrame(results['resource_availability'].items(), 
    #                     columns=['Resource', 'Availability']),
    #         x='Resource',
    #         y='Availability',
    #         color='Availability',
    #         color_continuous_scale='RdYlGn',
    #         title="Predicted Resource Availability"
    #     )
    #     st.plotly_chart(fig)
    #     st.markdown("""
    #         💡 **Chart Insight**: This chart shows predicted resource availability levels. 
    #         Green indicates good availability, while red suggests potential constraints.
    #     """)
        
    #     # Show recommendations
    #     # self._display_admission_recommendations(results)
    def _display_admission_predictions(self, results):
        """Display admission prediction results with interactive visualizations"""
        st.markdown("### Prediction Results")
    
        # Display key metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Length of Stay",
                f"{results['predicted_los']:.1f} days",
                f"{results['predicted_los'] - 4.5:.1f} vs target"
            )
            
        with col2:
            st.metric(
                "Expected Occupancy",
                f"{results['predicted_occupancy']:.1%}",
                f"{(results['predicted_occupancy'] - 0.85)*100:.1f}% vs target"
            )
            
        with col3:
            st.metric(
                "Optimal Admission Day",
                results['optimal_day'],
                "Recommended"
            )
        
        # Resource availability chart
        st.subheader("Resource Availability Forecast")
        
        # Ensure resource availability values are not negative
        resource_availability = results['resource_availability']
        resource_availability = {k: max(0, v) for k, v in resource_availability.items()}
        
        # Create the bar chart
        fig = px.bar(
            pd.DataFrame(resource_availability.items(), columns=['Resource', 'Availability']),
            x='Resource',
            y='Availability',
            color='Availability',
            color_continuous_scale='RdYlGn',
            title="Predicted Resource Availability"
        )
        
        st.plotly_chart(fig)
        
        st.markdown("""
            💡 **Chart Insight**: This chart shows predicted resource availability levels. 
            Green indicates good availability, while red suggests potential constraints.
        """)
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Developed by Vetapalem Vajralu</p>
            <p>Version 1.1.0 | Last Updated: January 2025</p>
            <small>This tool is for educational and research purposes only. 
        </div>
        """, unsafe_allow_html=True)
        
        # Optionally, show recommendations if needed
        # self._display_admission_recommendations(results)


    def _resource_forecasting_section(self):
        """Interactive resource forecasting section"""
        st.subheader("Resource Demand Forecasting")
        st.write("""
    This tool helps forecast the demand for critical resources such as beds, staff, ICU, and equipment. 
    The forecast is based on historical data, allowing users to visualize future resource needs and make informed decisions about resource allocation.
    """)
        
        # Time range selector
        forecast_range = st.slider(
            "Forecast Range (Days)",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # Resource type selector
        resource_type = st.selectbox(
            "Resource Type",
            ["Beds", "Staff", "ICU", "Equipment"]
        )
        
        if st.button("Generate Forecast"):
            # Generate and display forecast
            forecast_results = self._generate_resource_forecast(
                resource_type, 
                forecast_range
            )
            
            # Plot forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=self.filtered_data['Admission_Date'].tail(30),
                y=self.filtered_data[self._get_resource_column(resource_type)].tail(30),
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_results['dates'],
                y=forecast_results['values'],
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{resource_type} Demand Forecast",
                xaxis_title="Date",
                yaxis_title="Demand Level"
            )
            
            st.plotly_chart(fig)
            st.markdown("""
                💡 **Forecast Insight**: The dashed line shows predicted demand levels based on 
                historical patterns, seasonality, and current trends.
            """)
            
            # Display key insights
            self._display_forecast_insights(forecast_results)
    def _admission_forecasting_section(self):
        """Interactive Admission Forecasting Section"""
        st.subheader("Admission Forecasting")

        # Forecast Range Slider
        forecast_range = st.slider(
            "Forecast Range (Days)",
            min_value=1,
            max_value=30,
            value=7,
            key="forecast_range_slider"  # Unique key for this slider
        )

        # Generate Forecast Button
        if st.button("Generate Admission Forecast", key="generate_forecast_button"):  # Unique key for the button
            forecast_results = self._generate_admission_forecast(forecast_range)

            # Display forecast plot
            self._display_admission_forecast(forecast_results)
            

    def _generate_admission_forecast(self, forecast_range):
        """Generate admission forecast using historical data"""
        # Aggregate the data by date
        admission_data = self.filtered_data.groupby('Admission_Date').size()

        # Fit a basic ARIMA model to the historical data
        model = ARIMA(admission_data, order=(5, 1, 0))  # (p,d,q) order; adjust for your data
        model_fit = model.fit()

        # Forecast future admissions
        forecast_steps = forecast_range
        forecast_values = model_fit.forecast(steps=forecast_steps)

        # Add randomness for a non-linear appearance (introducing variability)
        forecast_values = [
            max(0, val + np.random.normal(scale=0.1 * val)) for val in forecast_values
        ]

        # Create a future date range for the forecast
        last_date = admission_data.index[-1] if isinstance(admission_data.index, pd.DatetimeIndex) else pd.to_datetime('2017-05-31')
        future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]

        # Prepare the forecast data
        forecast_data = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Admissions': forecast_values
        })

        return forecast_data
    def _display_admission_forecast(self, forecast_results):
        """Display the admission forecast with visualization"""
        # Enhanced Plot with variability
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast_results['Date'],
            y=forecast_results['Forecasted Admissions'],
            mode='lines+markers',
            name='Forecasted Admissions',
            line=dict(color='blue', dash='solid')
        ))

        # Add variability as a shaded area around the forecast line
        variability_range = 0.15 * forecast_results['Forecasted Admissions']
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_results['Date'], forecast_results['Date'][::-1]]),
            y=pd.concat([
                forecast_results['Forecasted Admissions'] - variability_range,
                (forecast_results['Forecasted Admissions'] + variability_range)[::-1]
            ]),
            fill='toself',
            fillcolor='rgba(0, 123, 255, 0.2)',  # Semi-transparent blue
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Forecast Variability'
        ))

        fig.update_layout(
            title="Predicted Number of Admissions with Variability",
            xaxis_title="Date",
            yaxis_title="Number of Admissions",
            template="plotly_white"
        )

        st.plotly_chart(fig)
        st.markdown("""
            💡 **Forecast Insight**: The shaded region around the forecast line indicates potential variability in 
            admissions based on historical trends and random fluctuations. This can help in planning resources 
            with some margin of safety.
        """)
        # Copyright Section
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Developed by Vetapalem Vajralu</p>
            <p>Version 1.1.0 | Last Updated: January 2025</p>
            <small>This tool is for educational and research purposes only. 
        </div>
        """, unsafe_allow_html=True)
    # def _generate_admission_forecast(self, forecast_range):
    #     """Generate probabilistic admission forecast using historical data."""
    #     # Aggregate the data by date
    #     admission_data = self.filtered_data.groupby('Admission_Date').size()

    #     # Calculate historical statistics
    #     historical_mean = admission_data.mean()
    #     historical_std = admission_data.std()

    #     # Fit an ARIMA model to historical data
    #     model = ARIMA(admission_data, order=(5, 1, 0))  # Example (p, d, q) order
    #     model_fit = model.fit()

    #     # Forecast future admissions
    #     forecast_steps = forecast_range
    #     forecast_values = model_fit.forecast(steps=forecast_steps)

    #     # Generate probability bins dynamically
    #     probability_bins = []
    #     for value in forecast_values:
    #         # Use historical standard deviation to create dynamic ranges
    #         low = max(0, value - 2 * historical_std)  # Lower bound (2 std dev below)
    #         mid_low = max(0, value - historical_std)  # 1 std dev below
    #         mid_high = value + historical_std         # 1 std dev above
    #         high = value + 2 * historical_std         # Upper bound (2 std dev above)

    #         # Assign dynamic probabilities
    #         probabilities = {
    #             f"{int(low)}-{int(mid_low)}": 0.2,  # Lower range
    #             f"{int(mid_low)}-{int(mid_high)}": 0.6,  # Mid-range (most likely)
    #             f"{int(mid_high)}-{int(high)}": 0.2  # Upper range
    #         }
    #         probability_bins.append(probabilities)

    #     # Generate future dates
    #     last_date = admission_data.index[-1] if isinstance(admission_data.index, pd.DatetimeIndex) else pd.to_datetime('2017-05-31')
    #     future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='D')[1:]

    #     # Prepare results
    #     forecast_data = pd.DataFrame({
    #         'Date': future_dates,
    #         'Probability Bins': probability_bins
    #     })

    #     return forecast_data
    # def _display_admission_forecast(self, forecast_results):
    #     """Display the admission forecast with probabilities."""
    #     st.markdown("### Admission Forecast with Probabilities")
        
    #     # Prepare data for visualization
    #     visualization_data = []
    #     for _, row in forecast_results.iterrows():
    #         date = row['Date']
    #         for bin_range, probability in row['Probability Bins'].items():
    #             visualization_data.append({
    #                 'Date': date,
    #                 'Admission Range': bin_range,
    #                 'Probability': probability
    #             })
        
    #     # Convert to DataFrame for Plotly
    #     viz_df = pd.DataFrame(visualization_data)

    #     # Create a stacked bar chart
    #     fig = px.bar(
    #         viz_df,
    #         x='Date',
    #         y='Probability',
    #         color='Admission Range',
    #         title="Admission Forecast Probabilities",
    #         labels={'Probability': 'Likelihood'},
    #         barmode='stack'
    #     )

    #     # Enhance layout
    #     fig.update_layout(
    #         xaxis_title="Date",
    #         yaxis_title="Probability",
    #         template="plotly_white",
    #         legend_title="Admission Range"
    #     )

    #     st.plotly_chart(fig)
    #     st.markdown("""
    #         💡 **Insight**: This chart shows the likelihood of different admission ranges over the forecast period. 
    #         Use this to plan resources for the most probable scenarios.
    #     """)

    def _capacity_planning_section(self):
        """Interactive capacity planning section"""
        st.subheader("Capacity Planning & Optimization")
        st.write("""
    Optimize hospital capacity by adjusting parameters like target occupancy rates and staff-to-patient ratios. 
    This feature helps in selecting wards and analyzing how changes affect hospital performance and resource requirements.
    """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Planning parameters
            target_occupancy = st.slider(
                "Target Occupancy Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05
            )
            
            staff_ratio = st.slider(
                "Target Staff-to-Patient Ratio",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
            
        with col2:
            selected_wards = st.multiselect(
                "Select Wards for Analysis",
                options=sorted(self.filtered_data['Ward'].unique()),
                default=sorted(self.filtered_data['Ward'].unique())[:3]
            )
        
        if st.button("Analyze Capacity"):
            # Generate capacity analysis
            capacity_analysis = self._analyze_capacity(
                target_occupancy,
                staff_ratio,
                selected_wards
            )
            
            # Display capacity analysis results
            self._display_capacity_analysis(capacity_analysis)
    
    def _calculate_resource_availability(self, ward, special_requirements):
        """Calculate resource availability based on historical data and current conditions"""
        ward_data = self.filtered_data[self.filtered_data['Ward'] == ward]
        
        availability = {
            'General Beds': 1 - ward_data['General_Bed_Occupancy'].mean(),
            'ICU Beds': 1 - ward_data['ICU_Bed_Occupancy'].mean(),
            'Staff': ward_data['Staff_Availability'].mean() / ward_data['Staff_Availability'].max(),
            'Equipment': 0.85  # Placeholder - could be calculated from actual equipment data
        }
        
        return availability

    def _generate_resource_forecast(self, resource_type, forecast_range):
        """Generate resource demand forecast"""
        # Get historical data for the resource
        historical_data = self.filtered_data[self._get_resource_column(resource_type)]
        
        # Generate dates for forecast
        last_date = self.filtered_data['Admission_Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_range
        )
        
        # Simple forecasting logic (could be enhanced with more sophisticated models)
        forecast_values = np.random.normal(
            historical_data.mean(),
            historical_data.std() * 0.5,
            size=forecast_range
        )
        
        return {
            'dates': forecast_dates,
            'values': forecast_values,
            'mean': historical_data.mean(),
            'peak': forecast_values.max(),
            'trough': forecast_values.min()
        }

    def _get_resource_column(self, resource_type):
        """Map resource type to corresponding data column"""
        resource_map = {
            'Beds': 'General_Bed_Occupancy',
            'Staff': 'Staff_Availability',
            'ICU': 'ICU_Bed_Occupancy',
            'Equipment': 'Daily_Bed_Availability'  # Placeholder
        }
        return resource_map[resource_type]

    # def _display_admission_recommendations(self, results):
    #     """Display actionable recommendations based on predictions"""
    #     st.markdown("### Recommendations")
        
    #     recommendations = []
        
    #     # Generate recommendations based on results
    #     if results['predicted_occupancy'] > 0.9:
    #         recommendations.append({
    #             'type': 'warning',
    #             'message': "High occupancy expected. Consider alternative admission dates."
    #         })
        
    #     if results['predicted_los'] > 5:
    #         recommendations.append({
    #             'type': 'info',
    #             'message': "Extended stay predicted. Prepare for longer-term resource allocation."
    #         })
        
    #     # Display recommendations with appropriate styling
    #     for rec in recommendations:
    #         st.markdown(f"""
    #             <div style='padding: 10px; background-color: {"red" if rec['type'] == "warning" else "blue"}; 
    #                         color: white; border-radius: 5px; margin: 5px;'>
    #                 {rec['message']}
    #             </div>
    #         """, unsafe_allow_html=True)

    def _display_forecast_insights(self, forecast_results):
        """Display key insights from forecast results"""
        st.markdown("### Key Forecast Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Demand",
                f"{forecast_results['mean']:.1f}",
                "Baseline"
            )
            
        with col2:
            st.metric(
                "Peak Demand",
                f"{forecast_results['peak']:.1f}",
                f"{(forecast_results['peak'] - forecast_results['mean']):.1f} vs avg"
            )
            
        with col3:
            st.metric(
                "Minimum Demand",
                f"{forecast_results['trough']:.1f}",
                f"{(forecast_results['trough'] - forecast_results['mean']):.1f} vs avg"
            )
        # Copyright Section
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Developed by Vetapalem Vajralu</p>
            <p>Version 1.1.0 | Last Updated: January 2025</p>
            <small>This tool is for educational and research purposes only. 
        </div>
        """, unsafe_allow_html=True)

    def _analyze_capacity(self, target_occupancy, staff_ratio, selected_wards):
        """Analyze capacity and generate optimization recommendations"""
        capacity_analysis = {}
        
        for ward in selected_wards:
            ward_data = self.filtered_data[self.filtered_data['Ward'] == ward]
            
            current_occupancy = ward_data['Occupancy_Rate'].mean()
            current_staff_ratio = ward_data['Staff_Per_Patient'].mean()
            
            capacity_analysis[ward] = {
                'current_occupancy': current_occupancy,
                'occupancy_gap': target_occupancy - current_occupancy,
                'current_staff_ratio': current_staff_ratio,
                'staff_ratio_gap': staff_ratio - current_staff_ratio,
                'optimization_potential': (1 - current_occupancy) * (1 - current_staff_ratio/staff_ratio)
            }
        
        return capacity_analysis

    def _display_capacity_analysis(self, capacity_analysis):
        """Display capacity analysis results with visualizations"""
        st.markdown("### Capacity Analysis Results")
        
        # Create DataFrame for visualization
        analysis_df = pd.DataFrame.from_dict(capacity_analysis, orient='index')
        
        # Plot occupancy comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current Occupancy',
            x=list(capacity_analysis.keys()),
            y=analysis_df['current_occupancy'],
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Occupancy Gap',
            x=list(capacity_analysis.keys()),
            y=analysis_df['occupancy_gap'],
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Ward Capacity Analysis",
            barmode='stack'
        )
        
        st.plotly_chart(fig)
        st.markdown("""
            💡 **Analysis Insight**: Blue bars show current occupancy levels, while red bars 
            indicate the gap to target occupancy. Taller red bars suggest more room for optimization.
        """)
        
        # Display optimization recommendations
        st.subheader("Optimization Recommendations")
        for ward, analysis in capacity_analysis.items():
            if analysis['optimization_potential'] > 0.1:
                st.markdown(f"""
                    <div style='padding: 10px; background-color: black; border-radius: 5px; margin: 5px;'>
                        <h4>{ward}</h4>
                        <p>Optimization potential: {analysis['optimization_potential']:.1%}</p>
                        <ul>
                            {'<li>Consider increasing bed capacity</li>' if analysis['occupancy_gap'] > 0 else ''}
                            {'<li>Staff allocation needs adjustment</li>' if abs(analysis['staff_ratio_gap']) > 0.1 else ''}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
        # Copyright Section
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Developed by Vetapalem Vajralu</p>
            <p>Version 1.1.0 | Last Updated: January 2025</p>
            <small>This tool is for educational and research purposes only. 
        </div>
        """, unsafe_allow_html=True)
    def metric_card(self, title, value, target):
        """Render a metric card"""
        st.markdown(f"""
            <div class="metric-container">
                <h3>{title}</h3>
                <h2>{value}</h2>
                <p>{target}</p>
            </div>
        """, unsafe_allow_html=True)
    
    def render_trends(self):
        """Render trend charts"""
        st.subheader("Occupancy Trends")
        
        # Create subplot with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Add occupancy trace
        fig.add_trace(
            go.Scatter(
                x=self.filtered_data['Admission_Date'],
                y=self.filtered_data['Occupancy_Rate'],
                name="Occupancy Rate"
            ),
            row=1, col=1
        )
        
        # Add staff availability trace
        fig.add_trace(
            go.Scatter(
                x=self.filtered_data['Admission_Date'],
                y=self.filtered_data['Staff_Per_Patient'],
                name="Staff per Patient"
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Hospital Operations Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_insights(self):
        """Render automated insights"""
        st.subheader("Key Insights")
        
        # Calculate insights
        avg_occupancy = self.filtered_data['Occupancy_Rate'].mean()
        peak_days = self.filtered_data.groupby('Day_of_Week')['Occupancy_Rate'].mean()
        busiest_day = peak_days.idxmax()
        
        st.markdown(f"""
            <div class="insight-box">
                <h4>Occupancy Insights</h4>
                <p>Average occupancy rate is {avg_occupancy:.1%} with highest occupancy on {busiest_day}s</p>
            </div>
        """, unsafe_allow_html=True)

def main():
    """Main entry point"""
    data_path = "data/hospitalfinal.csv"  # Replace with your data path
    dashboard = HospitalDashboard(data_path)
    dashboard.run()

if __name__ == "__main__":
    main()
