import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="SmartCart Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('smartcart_customers.csv')
    return df

@st.cache_data
def preprocess_data(df):
    """Data preprocessing pipeline"""
    df_clean = df.copy()
    
    # Fill missing Income
    df_clean["Income"] = df_clean["Income"].fillna(df_clean["Income"].mean())
    
    # Calculate Age
    df_clean["Age"] = 2026 - df_clean["Year_Birth"]
    
    # Customer Tenure
    df_clean["Dt_Customer"] = pd.to_datetime(df_clean["Dt_Customer"], dayfirst=True)
    reference_date = df_clean["Dt_Customer"].max()
    df_clean["Customer_Tenure_Days"] = (reference_date - df_clean["Dt_Customer"]).dt.days
    
    # Total Spending
    df_clean["Total_Spending"] = (df_clean["MntWines"] + df_clean["MntFruits"] + 
                                  df_clean["MntMeatProducts"] + df_clean["MntFishProducts"] + 
                                  df_clean["MntSweetProducts"] + df_clean["MntGoldProds"])
    
    # Total Children
    df_clean["Total_Children"] = df_clean["Kidhome"] + df_clean["Teenhome"]
    
    # Education Consolidation
    df_clean["Education"] = df_clean["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    
    # Living Status
    df_clean["Living_With"] = df_clean["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    
    # Drop unnecessary columns
    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", 
                    "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts", 
                    "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Remove outliers
    df_clean = df_clean[(df_clean["Age"] < 90) & (df_clean["Income"] < 600_000)]
    
    return df_clean

@st.cache_data
def encode_and_scale_data(df):
    """Encode categorical variables and scale data"""
    df_encoded = df.copy()
    
    # One-hot encode
    ohe = OneHotEncoder()
    cat_cols = ["Education", "Living_With"]
    enc_cols = ohe.fit_transform(df_encoded[cat_cols])
    enc_df = pd.DataFrame(enc_cols.toarray(), 
                         columns=ohe.get_feature_names_out(cat_cols), 
                         index=df_encoded.index)
    
    df_encoded = pd.concat([df_encoded.drop(columns=cat_cols), enc_df], axis=1)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    return X_scaled, df_encoded

@st.cache_data
def perform_pca(X_scaled):
    """Perform PCA for dimensionality reduction"""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return X_pca, pca, explained_variance

@st.cache_data
def find_optimal_clusters(X_pca):
    """Find optimal number of clusters using elbow and silhouette methods"""
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)
        
        labels = kmeans.labels_
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
    
    # Find elbow
    knee = KneeLocator(K_range, wcss, curve="convex", direction="decreasing")
    optimal_k = knee.elbow if knee.elbow else 4
    
    return wcss, silhouette_scores, optimal_k

@st.cache_data
def cluster_customers(X_pca, n_clusters=4):
    """Perform clustering"""
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca)
    
    # Agglomerative Clustering
    agg_clf = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_agg = agg_clf.fit_predict(X_pca)
    
    return labels_kmeans, labels_agg

# Main app
def main():
    # Header
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("🛒", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='header-title'>SmartCart Customer Analytics</div>", 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    df = load_data()
    df_clean = preprocess_data(df)
    X_scaled, df_encoded = encode_and_scale_data(df)
    X_pca, pca, explained_var = perform_pca(X_scaled)
    wcss, silhouette_scores, optimal_k = find_optimal_clusters(X_pca)
    labels_kmeans, labels_agg = cluster_customers(X_pca, n_clusters=optimal_k)
    
    # Add cluster labels to data
    df_clean['Cluster'] = labels_agg
    df_encoded['Cluster'] = labels_agg
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        page = st.radio("Select a page:", 
                       ["📈 Overview", "👥 Customer Segments", "📊 Detailed Analysis", 
                        "🔍 Clustering", "💡 Insights"])
    
    # Page: Overview
    if page == "📈 Overview":
        overview_page(df_clean, X_pca, labels_agg, optimal_k)
    
    # Page: Customer Segments
    elif page == "👥 Customer Segments":
        segments_page(df_clean, X_pca, labels_agg)
    
    # Page: Detailed Analysis
    elif page == "📊 Detailed Analysis":
        analysis_page(df_clean, df_encoded)
    
    # Page: Clustering
    elif page == "🔍 Clustering":
        clustering_page(X_pca, wcss, silhouette_scores, optimal_k, labels_agg)
    
    # Page: Insights
    elif page == "💡 Insights":
        insights_page(df_clean, labels_agg)

def overview_page(df_clean, X_pca, labels, n_clusters):
    """Overview page with key metrics"""
    st.markdown("### 📈 Dashboard Overview")
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df_clean), "+12%")
    with col2:
        st.metric("Avg Income", f"${df_clean['Income'].mean():.0f}", "+5%")
    with col3:
        st.metric("Avg Spending", f"${df_clean['Total_Spending'].mean():.0f}", "+8%")
    with col4:
        st.metric("Customer Segments", n_clusters, "Identified")
    
    st.markdown("---")
    
    # 3D Visualization
    st.markdown("### 🎯 Customer Distribution in 3D Space (PCA)")
    
    fig = go.Figure(data=[go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster")
        ),
        text=labels,
        hovertemplate='<b>Cluster: %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
    )])
    
    fig.update_layout(
        height=600,
        title_text="Customer Segments in 3D Space",
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Cluster Distribution")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_bar = go.Figure(data=[
            go.Bar(x=[f'Cluster {i}' for i in cluster_counts.index], 
                   y=cluster_counts.values,
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(cluster_counts)])
        ])
        fig_bar.update_layout(height=400, title_text="Customers per Cluster", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Income Distribution")
        fig_box = px.box(df_clean, y='Income', color='Cluster',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_clusters])
        fig_box.update_layout(height=400, title_text="Income by Cluster")
        st.plotly_chart(fig_box, use_container_width=True)

def segments_page(df_clean, X_pca, labels):
    """Customer segments analysis"""
    st.markdown("### 👥 Customer Segments Analysis")
    st.markdown("---")
    
    df_clean['Cluster'] = labels
    n_clusters = len(np.unique(labels))
    
    # Segment tabs
    tabs = st.tabs([f"Segment {i}" for i in range(n_clusters)])
    
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for cluster_id, tab in enumerate(tabs):
        with tab:
            cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
            
            st.markdown(f"#### <span style='color:{cluster_colors[cluster_id]}'>Segment {cluster_id}</span>", 
                       unsafe_allow_html=True)
            
            # Metrics for this segment
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Customers", len(cluster_data), f"{len(cluster_data)/len(df_clean)*100:.1f}%")
            with col2:
                st.metric("Avg Age", f"{cluster_data['Age'].mean():.1f} yrs")
            with col3:
                st.metric("Avg Income", f"${cluster_data['Income'].mean():.0f}")
            with col4:
                st.metric("Avg Spending", f"${cluster_data['Total_Spending'].mean():.0f}")
            
            st.markdown("---")
            
            # Characteristics
            st.markdown("**📋 Segment Characteristics:**")
            
            characteristics = f"""
            - **Average Age:** {cluster_data['Age'].mean():.1f} years
            - **Average Income:** ${cluster_data['Income'].mean():.0f}
            - **Average Total Spending:** ${cluster_data['Total_Spending'].mean():.0f}
            - **Average Tenure:** {cluster_data['Customer_Tenure_Days'].mean():.0f} days
            - **Most Common Education:** {cluster_data['Education'].mode()[0] if len(cluster_data) > 0 else 'N/A'}
            - **Most Common Living Status:** {cluster_data['Living_With'].mode()[0] if len(cluster_data) > 0 else 'N/A'}
            - **Average Children:** {cluster_data['Total_Children'].mean():.1f}
            - **Average Recency:** {cluster_data['Recency'].mean():.0f} days
            """
            
            st.markdown(characteristics)
            
            # Distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_income = px.histogram(cluster_data, x='Income', nbins=20,
                                        title=f"Income Distribution - Segment {cluster_id}",
                                        color_discrete_sequence=[cluster_colors[cluster_id]])
                fig_income.update_layout(height=300)
                st.plotly_chart(fig_income, use_container_width=True)
            
            with col2:
                fig_spending = px.histogram(cluster_data, x='Total_Spending', nbins=20,
                                          title=f"Spending Distribution - Segment {cluster_id}",
                                          color_discrete_sequence=[cluster_colors[cluster_id]])
                fig_spending.update_layout(height=300)
                st.plotly_chart(fig_spending, use_container_width=True)

def analysis_page(df_clean, df_encoded):
    """Detailed analysis page"""
    st.markdown("### 📊 Detailed Customer Analysis")
    st.markdown("---")
    
    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlations")
    
    # Select numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    corr_matrix = df_clean[numeric_cols].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmid=0
    ))
    fig_heatmap.update_layout(height=600, width=800)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plots
    st.markdown("#### 📈 Key Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter1 = px.scatter(df_clean, x='Income', y='Total_Spending',
                                 color='Age', hover_name='Cluster',
                                 title="Income vs Total Spending",
                                 color_continuous_scale='Viridis')
        fig_scatter1.update_layout(height=400)
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        fig_scatter2 = px.scatter(df_clean, x='Age', y='Total_Spending',
                                 color='Income', hover_name='Cluster',
                                 title="Age vs Total Spending",
                                 color_continuous_scale='Viridis')
        fig_scatter2.update_layout(height=400)
        st.plotly_chart(fig_scatter2, use_container_width=True)

def clustering_page(X_pca, wcss, silhouette_scores, optimal_k, labels):
    """Clustering analysis page"""
    st.markdown("### 🔍 Clustering Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Elbow Method")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(2, 11)), y=wcss,
                                       mode='lines+markers',
                                       name='WCSS'))
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red",
                           annotation_text=f"Optimal K={optimal_k}",
                           annotation_position="top right")
        fig_elbow.update_layout(height=400, title_text="WCSS vs Number of Clusters",
                               xaxis_title="Number of Clusters (K)",
                               yaxis_title="Within-Cluster Sum of Squares")
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        st.markdown("#### ⭐ Silhouette Score")
        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(x=list(range(2, 11)), y=silhouette_scores,
                                           mode='lines+markers',
                                           name='Silhouette Score',
                                           marker_color='#2ca02c'))
        fig_silhouette.add_vline(x=optimal_k, line_dash="dash", line_color="red",
                                annotation_text=f"K={optimal_k}",
                                annotation_position="top right")
        fig_silhouette.update_layout(height=400, title_text="Silhouette Score vs K",
                                     xaxis_title="Number of Clusters (K)",
                                     yaxis_title="Silhouette Score")
        st.plotly_chart(fig_silhouette, use_container_width=True)

def insights_page(df_clean, labels):
    """Key insights and recommendations"""
    st.markdown("### 💡 Strategic Insights & Recommendations")
    st.markdown("---")
    
    df_clean['Cluster'] = labels
    n_clusters = len(np.unique(labels))
    
    # Generate insights
    insights = []
    
    for cluster_id in range(n_clusters):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        
        avg_income = cluster_data['Income'].mean()
        avg_spending = cluster_data['Total_Spending'].mean()
        avg_age = cluster_data['Age'].mean()
        segment_size = len(cluster_data) / len(df_clean) * 100
        
        insight_title = ""
        insight_desc = ""
        
        if avg_income > df_clean['Income'].quantile(0.75) and avg_spending > df_clean['Total_Spending'].quantile(0.75):
            insight_title = f"🌟 Premium Customers (Segment {cluster_id})"
            insight_desc = f"High-value customers with excellent purchasing power. Focus on premium products and personalized service."
        elif avg_income < df_clean['Income'].quantile(0.25):
            insight_title = f"💰 Budget-Conscious (Segment {cluster_id})"
            insight_desc = f"Price-sensitive segment. Offer promotional deals and value packages."
        elif avg_age > 60:
            insight_title = f"👴 Senior Customers (Segment {cluster_id})"
            insight_desc = f"Older demographic. Ensure quality customer service and easy procurement."
        else:
            insight_title = f"👨‍👩‍👧‍👦 Family-Oriented (Segment {cluster_id})"
            insight_desc = f"Mixed income and spending. Offer family bundles and loyalty programs."
        
        insights.append({
            'title': insight_title,
            'description': insight_desc,
            'size': f"{segment_size:.1f}%",
            'income': f"${avg_income:.0f}",
            'spending': f"${avg_spending:.0f}",
            'age': f"{avg_age:.1f}"
        })
    
    # Display insights
    for insight in insights:
        with st.expander(insight['title']):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Size", insight['size'])
            with col2:
                st.metric("Income", insight['income'])
            with col3:
                st.metric("Spending", insight['spending'])
            with col4:
                st.metric("Avg Age", insight['age'])
            
            st.markdown(f"**Strategy:** {insight['description']}")
    
    st.markdown("---")
    st.markdown("### 🎯 General Recommendations")
    
    recommendations = """
    1. **Personalization**: Tailor marketing messages based on customer segment characteristics
    2. **Pricing Strategy**: Adjust pricing for different segments to maximize revenue
    3. **Product Mix**: Stock products preferred by each segment
    4. **Customer Retention**: Focus retention efforts on high-value customers
    5. **Cross-selling**: Identify opportunities to upsell complementary products
    6. **Loyalty Programs**: Create tiered rewards based on spending patterns
    7. **Communication**: Adjust communication frequency based on customer preference
    8. **Service Level**: Scale service quality based on customer segment value
    """
    
    st.markdown(recommendations)

if __name__ == "__main__":
    main()
