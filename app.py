import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load & Prepare Data ──────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('train.csv')

    # Drop high-missing columns
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.4].index.tolist()
    df.drop(columns=drop_cols, inplace=True)

    # Fill missing values
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature Engineering
    df['HouseAge']       = 2025 - df['YearBuilt']
    df['TotalSF']        = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    df['HasGarage']      = (df['GarageArea'] > 0).astype(int)
    df['HasPool']        = (df['PoolArea'] > 0).astype(int)

    # Encode
    df_model = df.copy()
    le = LabelEncoder()
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model.drop(columns=['Id', 'SalePrice'])
    y = np.log1p(df_model['SalePrice'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    gb  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    rf  = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    gb.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)

    gb_pred = gb.predict(X_test_s)
    rf_pred = rf.predict(X_test_s)

    metrics = {
        'Gradient Boosting': {
            'R2': r2_score(y_test, gb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred))
        },
        'Random Forest': {
            'R2': r2_score(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
        }
    }

    return df, X, gb, rf, scaler, X_test_s, y_test, gb_pred, rf_pred, metrics

df, X, gb_model, rf_model, scaler, X_test_s, y_test, gb_pred, rf_pred, metrics = load_and_train()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=80)
    st.title("🏠 House Price Predictor")
    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.info("**Dataset:** Ames Housing\n\n**Rows:** 1,460\n\n**Features:** 79\n\n**Goal:** Predict Sale Price")
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- 🐍 Python\n- 📊 Pandas\n- 🤖 Scikit-learn\n- 📈 Matplotlib\n- 🌐 Streamlit")
    st.markdown("---")
    st.caption("Made with ❤️ | Data Science Project")

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🏠 House Price Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Complete Machine Learning Project | Ames Housing Dataset</p>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🤖 Model Results", "🔮 Predict Price"])

# ── Tab 1: Data Overview ──────────────────────────────────────────────────────
with tab1:
    st.subheader("📂 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏠 Total Houses", f"{len(df):,}")
    col2.metric("📊 Features", f"{df.shape[1]-1}")
    col3.metric("💰 Avg Price", f"${df['SalePrice'].mean():,.0f}")
    col4.metric("📈 Max Price", f"${df['SalePrice'].max():,.0f}")

    st.markdown("---")
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Missing Values")
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False).head(10)
        if len(miss) > 0:
            st.bar_chart(miss)
        else:
            st.success("✅ No missing values after cleaning!")
    with col2:
        st.subheader("📊 Data Types")
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Type', 'Count']
        dtype_counts['Type'] = dtype_counts['Type'].astype(str)
        st.dataframe(dtype_counts, use_container_width=True)

# ── Tab 2: Visualizations ─────────────────────────────────────────────────────
with tab2:
    st.subheader("📈 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💰 SalePrice Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['SalePrice'], bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel('Sale Price ($)')
        ax.set_ylabel('Count')
        ax.set_title('Sale Price Distribution')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**📈 Log(SalePrice) Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log1p(df['SalePrice']), bins=50, color='coral', edgecolor='white')
        ax.set_xlabel('Log Sale Price')
        ax.set_ylabel('Count')
        ax.set_title('Log-Transformed Sale Price')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Correlation Bar Chart
    st.markdown("**🔗 Top Features Correlated with SalePrice**")
    num_cols = df.select_dtypes(include='number').columns
    corr = df[num_cols].corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr.values]
    ax.bar(corr.index, corr.values, color=colors)
    ax.set_xticklabels(corr.index, rotation=45, ha='right')
    ax.set_ylabel('Correlation')
    ax.set_title('Feature Correlation with SalePrice')
    ax.axhline(y=0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Scatter Plots
    st.markdown("**🔍 Feature vs SalePrice Scatter Plots**")
    selected_feat = st.selectbox("Select a Feature:", ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageArea', 'HouseAge', 'TotalBathrooms'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df[selected_feat], df['SalePrice'], alpha=0.4, color='#667eea')
    ax.set_xlabel(selected_feat)
    ax.set_ylabel('SalePrice ($)')
    ax.set_title(f'{selected_feat} vs SalePrice')
    st.pyplot(fig)
    plt.close()

# ── Tab 3: Model Results ──────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Model Performance")

    col1, col2 = st.columns(2)
    for i, (model_name, m) in enumerate(metrics.items()):
        col = col1 if i == 0 else col2
        with col:
            st.markdown(f"### {'🥇' if i==0 else '🥈'} {model_name}")
            c1, c2 = st.columns(2)
            c1.metric("R² Score", f"{m['R2']:.4f}", f"{m['R2']*100:.1f}%")
            c2.metric("RMSE (log)", f"{m['RMSE']:.4f}")

    st.markdown("---")

    # Actual vs Predicted
    st.markdown("**🎯 Actual vs Predicted Prices (Gradient Boosting)**")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y_actual = np.expm1(y_test)
    y_predicted = np.expm1(gb_pred)

    axes[0].scatter(y_actual, y_predicted, alpha=0.4, color='steelblue')
    lims = [min(y_actual.min(), y_predicted.min()), max(y_actual.max(), y_predicted.max())]
    axes[0].plot(lims, lims, 'r--', lw=2, label='Perfect Line')
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title('Actual vs Predicted')
    axes[0].legend()

    residuals = y_actual - y_predicted
    axes[1].scatter(y_predicted, residuals, alpha=0.4, color='coral')
    axes[1].axhline(y=0, color='black', linestyle='--')
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature Importance
    st.markdown("**🌲 Feature Importance (Random Forest)**")
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_imp = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('husl', 15)
    ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors[::-1])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Important Features')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 4: Predict Price ──────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Predict Your House Price")
    st.info("Fill in the details below and get an instant price prediction!")

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual  = st.slider("⭐ Overall Quality (1-10)", 1, 10, 7)
        gr_liv_area   = st.number_input("📐 Living Area (sq ft)", 500, 6000, 1500)
        total_sf      = st.number_input("🏗️ Total Square Footage", 500, 8000, 2000)
        garage_area   = st.number_input("🚗 Garage Area (sq ft)", 0, 1500, 400)

    with col2:
        year_built    = st.slider("📅 Year Built", 1870, 2025, 2000)
        total_baths   = st.slider("🛁 Total Bathrooms", 0.0, 6.0, 2.0, 0.5)
        fireplaces    = st.slider("🔥 Fireplaces", 0, 4, 1)
        bedroom       = st.slider("🛏️ Bedrooms", 0, 8, 3)

    with col3:
        garage_cars   = st.slider("🚙 Garage Capacity (cars)", 0, 4, 2)
        has_pool      = st.radio("🏊 Has Pool?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        neighborhood  = st.selectbox("🏘️ Neighborhood Type", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"])
        lot_area      = st.number_input("🌳 Lot Area (sq ft)", 1000, 50000, 8000)

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        house_age = 2025 - year_built

        # Create a sample input matching training features
        sample = pd.DataFrame(0, index=[0], columns=X.columns)

        # Fill known features
        feature_map = {
            'OverallQual': overall_qual,
            'GrLivArea': gr_liv_area,
            'TotalSF': total_sf,
            'GarageArea': garage_area,
            'GarageCars': garage_cars,
            'YearBuilt': year_built,
            'TotalBathrooms': total_baths,
            'Fireplaces': fireplaces,
            'BedroomAbvGr': bedroom,
            'HasPool': has_pool,
            'HasGarage': 1 if garage_area > 0 else 0,
            'HouseAge': house_age,
            'LotArea': lot_area,
            'FullBath': int(total_baths),
        }

        for feat, val in feature_map.items():
            if feat in sample.columns:
                sample[feat] = val

        sample_scaled = scaler.transform(sample)
        log_pred = gb_model.predict(sample_scaled)[0]
        predicted_price = np.expm1(log_pred)

        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎉 Predicted House Price</h2>
            <h1 style='font-size: 3rem; font-weight: 900;'>${predicted_price:,.0f}</h1>
            <p style='opacity: 0.9;'>Powered by Gradient Boosting Model | R² = {metrics['Gradient Boosting']['R2']:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🏠 House Age", f"{house_age} years")
        col2.metric("📐 Total Area", f"{total_sf:,} sq ft")
        col3.metric("⭐ Quality Score", f"{overall_qual}/10")
