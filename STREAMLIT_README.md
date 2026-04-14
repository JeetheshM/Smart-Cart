# SmartCart Analytics - Streamlit App

## 🛒 Overview

SmartCart Analytics is a comprehensive customer segmentation and analysis platform built with Streamlit. It provides actionable insights into customer behavior, spending patterns, and demographic characteristics using unsupervised learning algorithms.

## ✨ Features

- **📈 Dashboard Overview**: Key metrics and 3D customer distribution visualization
- **👥 Customer Segments**: Detailed analysis of 4 distinct customer segments with characteristics
- **📊 Detailed Analysis**: Feature correlations, scatter plots, and relationship analysis
- **🔍 Clustering Analysis**: Elbow method and Silhouette score analysis for optimal cluster selection
- **💡 Insights & Recommendations**: Strategic insights and actionable recommendations for each segment

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**
   ```bash
   streamlit run app.py
   ```

3. **Access the App**
   - Open your browser and go to `http://localhost:8501`

### File Requirements

Make sure the following files are in the same directory:
- `app.py` - Main Streamlit application
- `smartcart_customers.csv` - Customer dataset

## 📊 Data Processing

The app automatically performs:
- Data cleaning and missing value imputation
- Feature engineering (age, tenure, total spending calculation)
- Categorical variable encoding
- Data standardization and scaling
- PCA dimensionality reduction
- Customer clustering analysis

## 🧬 Machine Learning Pipeline

1. **Data Preprocessing**: Cleaning, encoding, and scaling
2. **Dimensionality Reduction**: PCA (3 components)
3. **Clustering**: K-means and Agglomerative Clustering
4. **Optimization**: Elbow method and Silhouette score analysis
5. **Segmentation**: Final customer segments with interpretable characteristics

## 📱 Streamlit Cloud Deployment

### Option 1: Direct Upload to Streamlit Cloud

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "SmartCart Streamlit App"
   git remote add origin https://github.com/JeetheshM/Smart-Cart.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Select the branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets (if needed)**
   - In Streamlit Cloud settings, add any API keys or secrets
   - They'll be available via `st.secrets`

### Option 2: Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Run locally:
```bash
docker build -t smartcart-app .
docker run -p 8501:8501 smartcart-app
```

## 👥 Customer Segments

The app identifies 4-5 distinct customer segments:

### Premium Customers
- High income and high spending
- **Strategy**: Premium products, personalized service

### Budget-Conscious
- Lower income, value-focused
- **Strategy**: Promotional deals, value packages

### Senior Customers
- Older demographic (60+)
- **Strategy**: Quality service, easy procurement

### Family-Oriented
- Mixed demographics, family-focused
- **Strategy**: Family bundles, loyalty programs

## 📈 Key Metrics Dashboard

- **Total Customers**: Count and growth percentage
- **Average Income**: Customer's average annual income
- **Average Spending**: Customer's average total spending
- **Customer Segments**: Number of identified segments

## 📊 Visualizations

- **3D PCA Plot**: Customer distribution in reduced dimensional space
- **Cluster Distribution**: Pie and bar charts for segment sizes
- **Income Distribution**: Box plots by segment
- **Spending Patterns**: Scatter plots of income vs spending
- **Feature Correlations**: Heatmap of feature relationships
- **Clustering Metrics**: Elbow curve and Silhouette scores

## 🔧 Customization

### Modify Cluster Count
In `app.py`, find the optimal_k calculation and adjust the clustering parameters:
```python
kmeans = KMeans(n_clusters=your_number, random_state=42, n_init=10)
```

### Add New Features
Update the `preprocess_data()` function to calculate new customer metrics or features.

### Customize Colors
Modify the `cluster_colors` list in the segment pages:
```python
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

## 📝 Data Dictionary

### Input Dataset: `smartcart_customers.csv`

| Column | Description |
|--------|-------------|
| ID | Customer identifier |
| Year_Birth | Birth year |
| Education | Education level |
| Marital_Status | Marital status |
| Income | Annual income |
| Recency | Recency of purchase |
| MntWines | Amount spent on wines |
| MntFruits | Amount spent on fruits |
| MntMeatProducts | Amount spent on meat |
| MntFishProducts | Amount spent on fish |
| MntSweetProducts | Amount spent on sweets |
| MntGoldProds | Amount spent on gold products |
| Kidhome | Number of kids at home |
| Teenhome | Number of teens at home |
| Dt_Customer | Customer joining date |
| Response | Campaign response |

### Generated Features

| Feature | Description |
|---------|-------------|
| Age | Calculated from birth year |
| Total_Spending | Sum of all product categories |
| Total_Children | Total kids + teens |
| Customer_Tenure_Days | Days since joining |
| Living_With | Consolidated marital status |

## 🐛 Troubleshooting

### App won't start
```bash
# Clear Streamlit cache
streamlit cache clear
streamlit run app.py
```

### Missing data files
- Ensure `smartcart_customers.csv` is in the same directory as `app.py`
- Check file encoding is UTF-8

### Import errors
```bash
pip install --upgrade -r requirements.txt
```

## 📧 Support

For issues or questions, please check the GitHub repository issues section or contact the development team.

## 📄 License

This project is part of the Prime Classes Unsupervised Learning course.

---

**Last Updated**: April 2026
**Version**: 1.0.0
