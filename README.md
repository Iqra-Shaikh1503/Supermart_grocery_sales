
# 🛒 Supermart Grocery Sales – Retail Analytics

## 📌 Project Overview

This project explores a fictional dataset representing grocery sales transactions from a supermarket in Tamil Nadu, India. The aim is to apply data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning to gain insights and predict sales trends, helping businesses make data-driven decisions.

## 📊 Objectives

- Clean and preprocess the dataset
- Perform exploratory data analysis (EDA)
- Visualize category-wise, city-wise, and time-based sales trends
- Build a machine learning model to predict sales
- Evaluate model performance
- Provide business insights and future recommendations

## 🧰 Tools & Technologies

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **SQL & Excel (for preliminary exploration)**

## 🧾 Dataset Details

- **Source:** [Google Drive Link](https://drive.google.com/file/d/1Vx-Ibn11HKofkJasjMZFyigemSu7TOeB/view)
- **Attributes:**
  - Order ID, Customer Name, Category, Sub Category
  - City, State, Region, Order Date
  - Sales, Discount, Profit
  - Extracted features: Month, Year, Month Number

## 📌 Key Steps

### 🔍 1. Data Preprocessing
- Handled missing values and duplicates
- Converted `Order Date` to datetime format
- Extracted `Day`, `Month`, `Year` for time-based analysis
- Label encoded categorical columns

### 📈 2. Exploratory Data Analysis (EDA)
- Analyzed sales trends by:
  - **Product Category**
  - **City**
  - **Month & Year**
- Generated visualizations using bar plots, line charts, pie charts, and heatmaps

### 🧠 3. Feature Engineering
- Derived meaningful features like `month_no`, `Month`, `year`
- Encoded regions and subcategories for ML input

### 🤖 4. Model Building
- Used **Linear Regression** to predict sales based on key features:
  - Category, Sub Category, Region, State, Month, Discount, Profit
- Model performance:
  - **R² Score:** 0.82
  - **Mean Squared Error:** ~1758.26

### 📊 5. Visualizations
- Sales over time
- Actual vs Predicted Sales
- Top 5 cities by sales
- Sales distribution by year and month

## 💡 Business Insights

- **Egg, Meat & Fish** category generated the highest revenue.
- Sales increased consistently over the months, suggesting effective seasonal strategies.
- **2017 and 2018** were peak revenue years.
- **Top 5 cities** contributed significantly to overall sales — crucial for targeted marketing.

## 🚀 Next Steps

- Apply **advanced ML models** like Random Forest, XGBoost
- Create a **Streamlit dashboard** for real-time sales prediction
- Integrate with **Power BI/Tableau** for business intelligence


## 🧠 Skills Demonstrated

- Data wrangling and cleaning
- Exploratory data analysis & visualization
- Supervised learning (regression)
- Business and retail analytics
- Feature engineering and model evaluation

## 👩‍💻 Author

Iqra Shaikh
Data Scientist | Machine Learning Enthusiast
📧 Contact: [shaikhiqra1503@gmail.com]
