# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from datetime import datetime

# Set page configuration with nature theme
st.set_page_config(
    page_title="Forest GHG Emissions Predictor",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external file
def load_css():
    try:
        with open('assets/style.css', 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")

# Apply CSS
load_css()

# Fix font issues for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Set matplotlib style to normal (remove dark theme config)
plt.style.use('default')
sns.set_palette(["#2d5016", "#4a7c3a", "#8db596", "#a3c585", "#8b7355"])

# Initialize session state
if 'model_history' not in st.session_state:
    st.session_state.model_history = []

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

if 'language' not in st.session_state:
    st.session_state.language = "en"

# Load translations
@st.cache_data
def load_translations():
    """Load translation files"""
    try:
        with open('translations/en.json', 'r', encoding='utf-8') as f:
            en_translations = json.load(f)
        with open('translations/pt.json', 'r', encoding='utf-8') as f:
            pt_translations = json.load(f)
        return {"en": en_translations, "pt": pt_translations}
    except FileNotFoundError:
        st.error("Translation files not found.")
        return {}

TRANSLATIONS = load_translations()

def get_text(category, key):
    """Helper function to get text in current language"""
    try:
        return TRANSLATIONS[st.session_state.language][category][key]
    except KeyError:
        return f"[{category}.{key}]"

def create_navbar():
    """Create a top navigation bar with language switcher"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            btn_style = "primary" if st.session_state.current_page == "main" else "secondary"
            if st.button("🌲 " + get_text("navigation", "main"), width='stretch', type=btn_style):
                st.session_state.current_page = "main"
                st.rerun()
        
        with nav_col2:
            btn_style = "primary" if st.session_state.current_page == "history" else "secondary"
            if st.button("📊 " + get_text("navigation", "history"), width='stretch', type=btn_style):
                st.session_state.current_page = "history"
                st.rerun()
        
        with nav_col3:
            btn_style = "primary" if st.session_state.current_page == "visualizations" else "secondary"
            if st.button("📈 " + get_text("navigation", "visualizations"), width='stretch', type=btn_style):
                st.session_state.current_page = "visualizations"
                st.rerun()
    
    with col2:
        st.markdown('<div class="language-switcher">', unsafe_allow_html=True)
        if st.session_state.language == "en":
            if st.button("🇧🇷 PT", width='stretch', help="Switch to Portuguese"):
                st.session_state.language = "pt"
                st.rerun()
        else:
            if st.button("🇺🇸 EN", width='stretch', help="Mudar para Inglês"):
                st.session_state.language = "en"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def create_header(title, description):
    """Create a styled header section"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the datasets"""
    try:
        pxp_df = pd.read_csv("data/ExioML_factor_accounting_PxP.csv")
        ixi_df = pd.read_csv("data/ExioML_factor_accounting_IxI.csv")
        return pxp_df, ixi_df
    except FileNotFoundError:
        st.error(get_text("common", "data_files_not_found"))
        return None, None

def preprocess_data(data):
    """Apply the same preprocessing as in your notebook"""
    data_log = data.copy()
    columns_to_transform = [
        'Value Added [M.EUR]', 
        'Employment [1000 p.]', 
        'GHG emissions [kg CO2 eq.]', 
        'Energy Carrier Net Total [TJ]'
    ]
    
    for column in columns_to_transform:
        if column in data_log.columns:
            data_log[column] = np.log1p(data_log[column])
    
    return data_log

def safe_dataframe_display(df, max_rows=1000):
    """Safely display dataframe by converting object dtypes to string"""
    display_df = df.copy()
    
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].astype(str)
    
    if len(display_df) > max_rows:
        st.info(f"Showing first {max_rows} rows of {len(display_df)}")
        display_df = display_df.head(max_rows)
    
    return display_df

def train_and_evaluate_model(data_log, hyperparams, dataset_name):
    """Train RandomForest model with given hyperparameters"""
    features = data_log.drop(columns=['region', 'sector', 'GHG emissions [kg CO2 eq.]'])
    target = data_log['GHG emissions [kg CO2 eq.]']
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    rf_regressor = RandomForestRegressor(**hyperparams)
    rf_regressor.fit(X_train, y_train)
    
    y_pred_test = rf_regressor.predict(X_test)
    y_pred_train = rf_regressor.predict(X_train)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    importances = rf_regressor.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    model_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': dataset_name,
        'hyperparameters': hyperparams.copy(),
        'metrics': {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'feature_importance': feature_importance_df,
        'model': rf_regressor,
        'features': features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    st.session_state.model_history.append(model_record)
    st.session_state.current_model = model_record
    
    return model_record

def display_model_results(results):
    """Display model results in a standardized way"""
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("📊 " + get_text("model_results", "model_performance"))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Train MSE", f"{results['metrics']['train_mse']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test MSE", f"{results['metrics']['test_mse']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Train R²", f"{results['metrics']['train_r2']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test R²", f"{results['metrics']['test_r2']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("🔍 " + get_text("model_results", "feature_importance"))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = results['feature_importance'].head(10)
        
        # Fixed seaborn barplot with proper parameters
        colors_list = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_features))).tolist()
        sns.barplot(
            data=top_features, 
            x='Importance', 
            y='Feature', 
            hue='Feature',
            ax=ax, 
            palette=colors_list,
            legend=False,
            saturation=0.8
        )
        
        ax.set_title("Top 10 Most Important Features", fontsize=14, pad=20)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        st.pyplot(fig)
    
    with col2:
        display_df = safe_dataframe_display(results['feature_importance'].head(10))
        st.dataframe(display_df, width='content')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("📈 " + get_text("model_results", "actual_vs_predicted"))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set plot - normal styling
    ax1.scatter(results['y_train'], results['y_pred_train'], alpha=0.6, color='#2d5016')
    ax1.plot([results['y_train'].min(), results['y_train'].max()], 
            [results['y_train'].min(), results['y_train'].max()], 'r--', lw=2, alpha=0.8)
    ax1.set_xlabel(get_text("model_results", "actual"))
    ax1.set_ylabel(get_text("model_results", "predicted"))
    ax1.set_title("Training Set")
    ax1.grid(True, alpha=0.3)
    
    # Test set plot - normal styling
    ax2.scatter(results['y_test'], results['y_pred_test'], alpha=0.6, color='#4a7c3a')
    ax2.plot([results['y_test'].min(), results['y_test'].max()], 
            [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2, alpha=0.8)
    ax2.set_xlabel(get_text("model_results", "actual"))
    ax2.set_ylabel(get_text("model_results", "predicted"))
    ax2.set_title("Test Set")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("⚙️ " + get_text("model_results", "hyperparameters_used"))
    st.json(results['hyperparameters'])
    st.markdown('</div>', unsafe_allow_html=True)

def main_page():
    """Main training page"""
    create_header(
        get_text("main_page", "title"),
        get_text("main_page", "description")
    )
    
    pxp_df, ixi_df = load_data()
    if pxp_df is None or ixi_df is None:
        return
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 " + get_text("sidebar", "dataset_selection"))
    dataset_choice = st.sidebar.radio(
        get_text("sidebar", "choose_dataset"),
        ["PxP (Process x Process)", "IxI (Industry x Industry)"],
        key="dataset_radio",
        help=f"{get_text('sidebar', 'pxp_description')}, {get_text('sidebar', 'ixi_description')}"
    )
    
    selected_data = pxp_df if dataset_choice == "PxP (Process x Process)" else ixi_df
    dataset_name = "PxP" if dataset_choice == "PxP (Process x Process)" else "IxI"
    
    st.subheader("📊 " + get_text("main_page", "dataset_overview"))
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f8fff0 0%, #f0f7e6 100%); 
                    padding: 1rem; border-radius: 10px; border-left: 4px solid #4a7c3a;'>
            <h4 style='color: #2d5016; margin: 0;'>Dataset Information</h4>
            <p style='color: #2d5016; margin: 0.5rem 0;'><strong>Selected:</strong> {dataset_choice}</p>
            <p style='color: #2d5016; margin: 0;'><strong>Shape:</strong> {selected_data.shape}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.checkbox("📋 " + get_text("main_page", "show_raw_data"), key="show_raw_data"):
            display_data = safe_dataframe_display(selected_data)
            st.dataframe(display_data, width='content')
    st.markdown('</div>', unsafe_allow_html=True)
    
    data_log = preprocess_data(selected_data)

    st.sidebar.markdown("### 🎛️ " + get_text("sidebar", "hyperparameter_tuning"))

    n_estimators = st.sidebar.slider(
        get_text("sidebar", "n_estimators_label"),
        min_value=10, max_value=500, value=100, step=10,
        key="n_estimators", help=get_text("sidebar", "n_estimators_help")
    )

    max_depth_none = st.sidebar.checkbox(
        get_text("sidebar", "unlimited_depth"), value=False,
        key="unlimited_depth", help=get_text("sidebar", "unlimited_depth_help")
    )

    if max_depth_none:
        max_depth = None
        st.sidebar.success(get_text("common", "max_depth_set"))
    else:
        max_depth = st.sidebar.slider(
            get_text("sidebar", "max_depth_label"), min_value=1, max_value=50, value=10,
            key="max_depth", help=get_text("sidebar", "max_depth_help")
        )

    min_samples_split = st.sidebar.slider(
        get_text("sidebar", "min_samples_split"), min_value=2, max_value=20, value=2,
        key="min_samples_split", help=get_text("sidebar", "min_samples_split_help")
    )

    min_samples_leaf = st.sidebar.slider(
        get_text("sidebar", "min_samples_leaf"), min_value=1, max_value=10, value=1,
        key="min_samples_leaf", help=get_text("sidebar", "min_samples_leaf_help")
    )

    max_features = st.sidebar.selectbox(
        get_text("sidebar", "max_features"), options=['sqrt', 'log2', None], index=0,
        key="max_features", help=get_text("sidebar", "max_features_help")
    )

    bootstrap = st.sidebar.checkbox(
        get_text("sidebar", "bootstrap"), value=True,
        key="bootstrap", help=get_text("sidebar", "bootstrap_help")
    )
    
    hyperparams = {
        'n_estimators': n_estimators, 'max_depth': max_depth,
        'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
        'max_features': max_features, 'bootstrap': bootstrap,
        'random_state': 42, 'n_jobs': -1
    }
    
    if st.sidebar.button("🚀 " + get_text("sidebar", "train_model"), type="primary", key="train_button"):
        with st.spinner(get_text("common", "training_model")):
            results = train_and_evaluate_model(data_log, hyperparams, dataset_name)
            display_model_results(results)

    if st.session_state.current_model:
        st.sidebar.success(f"✅ {get_text('sidebar', 'last_trained')}: {st.session_state.current_model['timestamp']}")

def history_page():
    """Model history page"""
    create_header(
        get_text("history_page", "title"),
        get_text("history_page", "description")
    )
    
    if not st.session_state.model_history:
        st.info(get_text("history_page", "no_models"))
        return
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("📋 " + get_text("history_page", "training_history"))
    
    history_data = []
    for i, model in enumerate(st.session_state.model_history):
        history_data.append({
            'Model ID': i + 1, 'Timestamp': model['timestamp'], 'Dataset': model['dataset'],
            'n_estimators': model['hyperparameters']['n_estimators'],
            'max_depth': str(model['hyperparameters']['max_depth']),
            'Train MSE': f"{model['metrics']['train_mse']:.4f}",
            'Test MSE': f"{model['metrics']['test_mse']:.4f}",
            'Train R²': f"{model['metrics']['train_r2']:.4f}",
            'Test R²': f"{model['metrics']['test_r2']:.4f}"
        })
    
    history_df = pd.DataFrame(history_data)
    for col in history_df.columns:
        history_df[col] = history_df[col].astype(str)
    
    st.dataframe(history_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("🔍 " + get_text("history_page", "detailed_view"))
    model_options = [f"Model {i+1} - {model['timestamp']} - {model['dataset']}" 
                    for i, model in enumerate(st.session_state.model_history)]
    
    selected_model_idx = st.selectbox(
        get_text("history_page", "select_model"), range(len(model_options)),
        format_func=lambda x: model_options[x], key="model_selector"
    )
    
    if selected_model_idx is not None:
        selected_model = st.session_state.model_history[selected_model_idx]
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance",
            "Hyperparameters", 
            "Feature Importance", 
            "Actual vs Predicted"
        ])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric(get_text("model_results", "train_mse"), f"{selected_model['metrics']['train_mse']:.4f}")
            with col2: st.metric(get_text("model_results", "test_mse"), f"{selected_model['metrics']['test_mse']:.4f}")
            with col3: st.metric(get_text("model_results", "train_r2"), f"{selected_model['metrics']['train_r2']:.4f}")
            with col4: st.metric(get_text("model_results", "test_r2"), f"{selected_model['metrics']['test_r2']:.4f}")
            
            st.write(f"**Dataset:** {selected_model['dataset']}")
            st.write(f"**Training Time:** {selected_model['timestamp']}")
        
        with tab2:
            st.subheader("Hyperparameters")
            st.json(selected_model['hyperparameters'])
        
        with tab3:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = selected_model['feature_importance'].head(10)
            
            colors_list = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_features))).tolist()
            sns.barplot(
                data=top_features, 
                x='Importance', 
                y='Feature', 
                hue='Feature',
                ax=ax, 
                palette=colors_list,
                legend=False,
                saturation=0.8
            )
            ax.set_title("Top 10 Most Important Features")
            st.pyplot(fig)
            st.dataframe(selected_model['feature_importance'], width='content')
        
        with tab4:
            st.subheader("Actual vs Predicted Values")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(selected_model['y_train'], selected_model['y_pred_train'], alpha=0.5, color='#2d5016')
            ax1.plot([selected_model['y_train'].min(), selected_model['y_train'].max()], 
                    [selected_model['y_train'].min(), selected_model['y_train'].max()], 'r--', lw=2)
            ax1.set_xlabel(get_text("model_results", "actual"))
            ax1.set_ylabel(get_text("model_results", "predicted"))
            ax1.set_title("Training Set")
            ax1.grid(True, alpha=0.3)
            
            ax2.scatter(selected_model['y_test'], selected_model['y_pred_test'], alpha=0.5, color='#4a7c3a')
            ax2.plot([selected_model['y_test'].min(), selected_model['y_test'].max()], 
                    [selected_model['y_test'].min(), selected_model['y_test'].max()], 'r--', lw=2)
            ax2.set_xlabel(get_text("model_results", "actual"))
            ax2.set_ylabel(get_text("model_results", "predicted"))
            ax2.set_title("Test Set")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.sidebar.header(get_text("sidebar", "management"))
    if st.sidebar.button(get_text("sidebar", "clear_history"), key="clear_history"):
        st.session_state.model_history = []
        st.session_state.current_model = None
        st.rerun()

def visualizations_page():
    """Dedicated visualizations page"""
    create_header(
        get_text("visualizations_page", "title"),
        get_text("visualizations_page", "description")
    )
    
    pxp_df, ixi_df = load_data()
    if pxp_df is None or ixi_df is None:
        return
    
    st.sidebar.header(get_text("sidebar", "dataset_selection"))
    dataset_choice = st.sidebar.radio(
        get_text("sidebar", "choose_dataset"),
        ["PxP (Process x Process)", "IxI (Industry x Industry)"],
        key="viz_dataset_radio",
        help=f"{get_text('sidebar', 'pxp_description')}, {get_text('sidebar', 'ixi_description')}"
    )
    
    selected_data = pxp_df if dataset_choice == "PxP (Process x Process)" else ixi_df
    
    st.markdown('<div class="forest-card">', unsafe_allow_html=True)
    st.subheader("📈 " + get_text("main_page", "dataset_overview"))
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**{get_text('main_page', 'selected_dataset')}:** {dataset_choice}")
        st.write(f"**{get_text('main_page', 'shape')}:** {selected_data.shape}")
    
    with col2:
        st.write(f"**{get_text('visualizations_page', 'num_features')}:** {len(selected_data.columns)}")
        st.write(f"**{get_text('visualizations_page', 'num_samples')}:** {len(selected_data)}")
    
    with col3:
        if st.checkbox(get_text("visualizations_page", "show_info"), key="viz_show_info"):
            st.write(f"**{get_text('visualizations_page', 'columns')}:**")
            for col in selected_data.columns:
                st.write(f"- {col}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    data_log = preprocess_data(selected_data)
    
    tab1, tab2, tab3 = st.tabs([
        "Data Distributions", 
        "Correlation Matrix", 
        "Raw Data"
    ])
    
    with tab1:
        st.markdown('<div class="forest-card">', unsafe_allow_html=True)
        st.subheader("Log-Transformed Data Distributions")
        
        columns_to_transform = [
            'Value Added [M.EUR]', 'Employment [1000 p.]', 
            'GHG emissions [kg CO2 eq.]', 'Energy Carrier Net Total [TJ]'
        ]
        
        existing_columns = [col for col in columns_to_transform if col in data_log.columns]
        
        if not existing_columns:
            st.warning(get_text("visualizations_page", "no_transformable"))
        else:
            n_cols = 2
            n_rows = (len(existing_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, column in enumerate(existing_columns):
                if i < len(axes):
                    sns.histplot(data_log[column], kde=True, ax=axes[i], color='#2d5016')
                    axes[i].set_title(f'Log-Transformed {column}')
                    axes[i].set_xlabel(column)
                    axes[i].set_ylabel('Frequency')
            
            for i in range(len(existing_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Descriptive Statistics")
            stats_df = safe_dataframe_display(data_log[existing_columns].describe())
            st.dataframe(stats_df, width='content')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="forest-card">', unsafe_allow_html=True)
        st.subheader("Correlation Matrix")
        
        columns_to_transform = [
            'Value Added [M.EUR]', 'Employment [1000 p.]', 
            'GHG emissions [kg CO2 eq.]', 'Energy Carrier Net Total [TJ]'
        ]
        
        existing_columns = [col for col in columns_to_transform if col in data_log.columns]
        
        if len(existing_columns) < 2:
            st.warning(get_text("visualizations_page", "need_columns"))
        else:
            correlation_matrix = data_log[existing_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title("Correlation Matrix of Log-Transformed Data")
            st.pyplot(fig)
            
            st.subheader("Correlation Insights")
            
            corr_matrix = correlation_matrix.copy()
            np.fill_diagonal(corr_matrix.values, 0)
            
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_value
                        })
            
            if strong_correlations:
                st.write("**Strong Correlations (|r| > 0.5):**")
                strong_corr_df = pd.DataFrame(strong_correlations)
                strong_corr_df = strong_corr_df.astype(str)
                st.dataframe(strong_corr_df.sort_values('Correlation', key=lambda x: x.astype(float).abs(), ascending=False), width='content')
            else:
                st.info(get_text("visualizations_page", "no_strong_correlations"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="forest-card">', unsafe_allow_html=True)
        st.subheader("Raw Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox(get_text("visualizations_page", "show_raw"), key="viz_show_raw"):
                display_data = safe_dataframe_display(selected_data)
                st.dataframe(display_data, width='content')
        
        with col2:
            if st.checkbox(get_text("visualizations_page", "show_log"), key="viz_show_log"):
                display_data = safe_dataframe_display(data_log)
                st.dataframe(display_data, width='content')
        
        st.subheader("Data Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Data Types:**")
            dtype_info = selected_data.dtypes.reset_index()
            dtype_info.columns = ['Column', 'Data Type']
            dtype_info['Data Type'] = dtype_info['Data Type'].astype(str)
            st.dataframe(dtype_info, hide_index=True, width='content')
        
        with info_col2:
            st.write("**Missing Values:**")
            missing_info = selected_data.isnull().sum().reset_index()
            missing_info.columns = ['Column', 'Missing Values']
            missing_info = missing_info[missing_info['Missing Values'] > 0]
            if len(missing_info) > 0:
                missing_info = missing_info.astype(str)
                st.dataframe(missing_info, hide_index=True, width='content')
            else:
                st.success(get_text("visualizations_page", "no_missing"))
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    create_navbar()
    
    if st.session_state.current_page == "main":
        main_page()
    elif st.session_state.current_page == "history":
        history_page()
    elif st.session_state.current_page == "visualizations":
        visualizations_page()

if __name__ == "__main__":
    main()