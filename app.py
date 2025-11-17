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

# Set page configuration
st.set_page_config(
    page_title="RandomForest GHG Emissions Predictor",
    page_icon="🌳",
    layout="wide"
)

# Initialize session state for model history and navigation
if 'model_history' not in st.session_state:
    st.session_state.model_history = []

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

def create_navbar():
    """Create a top navigation bar"""
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            justify-content: center;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .navbar a {
            margin: 0 15px;
            text-decoration: none;
            color: #31333F;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .navbar a:hover {
            background-color: #4e8cff;
            color: white;
        }
        .navbar a.active {
            background-color: #4e8cff;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create navbar with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Main", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "main" else "secondary"):
            st.session_state.current_page = "main"
            st.rerun()
    
    with col2:
        if st.button("📊 History", use_container_width=True,
                    type="primary" if st.session_state.current_page == "history" else "secondary"):
            st.session_state.current_page = "history"
            st.rerun()
    
    with col3:
        if st.button("📈 Visualizations", use_container_width=True,
                    type="primary" if st.session_state.current_page == "visualizations" else "secondary"):
            st.session_state.current_page = "visualizations"
            st.rerun()

@st.cache_data
def load_data():
    """Load and cache the datasets"""
    try:
        pxp_df = pd.read_csv("data/ExioML_factor_accounting_PxP.csv")
        ixi_df = pd.read_csv("data/ExioML_factor_accounting_IxI.csv")
        return pxp_df, ixi_df
    except FileNotFoundError:
        st.error("Data files not found. Please make sure the CSV files are in the 'data' folder.")
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
    
    # Log transformation
    for column in columns_to_transform:
        if column in data_log.columns:
            data_log[column] = np.log1p(data_log[column])
    
    return data_log

def train_and_evaluate_model(data_log, hyperparams, dataset_name):
    """Train RandomForest model with given hyperparameters"""
    # Prepare features and target
    features = data_log.drop(columns=['region', 'sector', 'GHG emissions [kg CO2 eq.]'])
    target = data_log['GHG emissions [kg CO2 eq.]']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Train model
    rf_regressor = RandomForestRegressor(**hyperparams)
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = rf_regressor.predict(X_test)
    y_pred_train = rf_regressor.predict(X_train)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Feature importance
    importances = rf_regressor.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Create model record
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
    
    # Add to history
    st.session_state.model_history.append(model_record)
    st.session_state.current_model = model_record
    
    return model_record

def display_model_results(results):
    """Display model results in a standardized way"""
    # Display results
    st.subheader("📊 Model Performance")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train MSE", f"{results['metrics']['train_mse']:.4f}")
    with col2:
        st.metric("Test MSE", f"{results['metrics']['test_mse']:.4f}")
    with col3:
        st.metric("Train R²", f"{results['metrics']['train_r2']:.4f}")
    with col4:
        st.metric("Test R²", f"{results['metrics']['test_r2']:.4f}")
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = results['feature_importance'].head(10)
        sns.barplot(data=top_features, x='Importance', y='Feature', ax=ax)
        ax.set_title('Top 10 Most Important Features')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Features')
        st.pyplot(fig)
    
    with col2:
        st.dataframe(results['feature_importance'].head(10))
    
    # Actual vs Predicted plot
    st.subheader("📈 Actual vs Predicted Values")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set
    ax1.scatter(results['y_train'], results['y_pred_train'], alpha=0.5)
    ax1.plot([results['y_train'].min(), results['y_train'].max()], 
            [results['y_train'].min(), results['y_train'].max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Training Set')
    
    # Test set
    ax2.scatter(results['y_test'], results['y_pred_test'], alpha=0.5)
    ax2.plot([results['y_test'].min(), results['y_test'].max()], 
            [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Test Set')
    
    st.pyplot(fig)
    
    # Current hyperparameters
    st.subheader("⚙️ Hyperparameters Used")
    st.json(results['hyperparameters'])

def main_page():
    """Main training page"""
    st.title("🌳 RandomForest GHG Emissions Predictor")
    st.markdown("""
    This app allows you to tune hyperparameters for the RandomForest regressor 
    that predicts GHG emissions based on economic and energy data.
    """)
    
    # Load data
    pxp_df, ixi_df = load_data()
    
    if pxp_df is None or ixi_df is None:
        return
    
    # Dataset selection
    st.sidebar.header("📊 Dataset Selection")
    dataset_choice = st.sidebar.radio(
        "Choose dataset:",
        ["PxP (Process x Process)", "IxI (Industry x Industry)"],
        key="dataset_radio",
        help="PxP: Process-level data, IxI: Industry-level data"
    )
    
    selected_data = pxp_df if dataset_choice == "PxP (Process x Process)" else ixi_df
    dataset_name = "PxP" if dataset_choice == "PxP (Process x Process)" else "IxI"
    
    # Display dataset info
    st.subheader("📈 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Selected Dataset:** {dataset_choice}")
        st.write(f"**Shape:** {selected_data.shape}")
    
    with col2:
        if st.checkbox("Show raw data", key="show_raw_data"):
            st.dataframe(selected_data.head())
    
    # Preprocess data
    data_log = preprocess_data(selected_data)

    # Hyperparameter tuning sidebar
    st.sidebar.header("🎛️ Hyperparameter Tuning")

    n_estimators = st.sidebar.slider(
        "Number of Trees (n_estimators)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="n_estimators",
        help="Number of trees in the forest"
    )

    max_depth_none = st.sidebar.checkbox(
        "Unlimited Max Depth (None)",
        value=False,
        key="unlimited_depth",
        help="If checked, trees will grow until all leaves are pure"
    )

    if max_depth_none:
        max_depth = None
        st.sidebar.info("Max Depth set to None (unlimited)")
    else:
        max_depth = st.sidebar.slider(
            "Max Depth",
            min_value=1,
            max_value=50,
            value=10,
            key="max_depth",
            help="Maximum depth of the tree"
        )

    min_samples_split = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        key="min_samples_split",
        help="Minimum number of samples required to split an internal node"
    )

    min_samples_leaf = st.sidebar.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=10,
        value=1,
        key="min_samples_leaf",
        help="Minimum number of samples required to be at a leaf node"
    )

    max_features = st.sidebar.selectbox(
        "Max Features",
        options=['sqrt', 'log2', None],
        index=0,
        key="max_features",
        help="Number of features to consider when looking for the best split"
    )

    bootstrap = st.sidebar.checkbox(
        "Bootstrap",
        value=True,
        key="bootstrap",
        help="Whether bootstrap samples are used when building trees"
    )
    
    # Hyperparameters dictionary
    hyperparams = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model button
    if st.sidebar.button("🚀 Train Model", type="primary", key="train_button"):
        with st.spinner("Training model... This may take a few moments."):
            results = train_and_evaluate_model(data_log, hyperparams, dataset_name)
            display_model_results(results)

    # Display current model if exists
    if st.session_state.current_model:
        st.sidebar.success(f"Last trained: {st.session_state.current_model['timestamp']}")

def history_page():
    """Model history page"""
    st.title("📊 Model History")
    st.markdown("""
    View all previously trained models, their hyperparameters, and performance metrics.
    """)
    
    if not st.session_state.model_history:
        st.info("No models have been trained yet. Go to the main page to train your first model!")
        return
    
    # Display summary table
    st.subheader("📋 Training History Summary")
    
    # Create summary dataframe
    history_data = []
    for i, model in enumerate(st.session_state.model_history):
        history_data.append({
            'Model ID': i + 1,
            'Timestamp': model['timestamp'],
            'Dataset': model['dataset'],
            'n_estimators': model['hyperparameters']['n_estimators'],
            'max_depth': model['hyperparameters']['max_depth'],
            'Train MSE': f"{model['metrics']['train_mse']:.4f}",
            'Test MSE': f"{model['metrics']['test_mse']:.4f}",
            'Train R²': f"{model['metrics']['train_r2']:.4f}",
            'Test R²': f"{model['metrics']['test_r2']:.4f}"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Model selection for detailed view
    st.subheader("🔍 Detailed Model View")
    model_options = [f"Model {i+1} - {model['timestamp']} - {model['dataset']}" 
                    for i, model in enumerate(st.session_state.model_history)]
    
    selected_model_idx = st.selectbox(
        "Select a model to view details:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        key="model_selector"
    )
    
    if selected_model_idx is not None:
        selected_model = st.session_state.model_history[selected_model_idx]
        
        # Display model details in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "⚙️ Hyperparameters", "📈 Feature Importance", "🔄 Actual vs Predicted"])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train MSE", f"{selected_model['metrics']['train_mse']:.4f}")
            with col2:
                st.metric("Test MSE", f"{selected_model['metrics']['test_mse']:.4f}")
            with col3:
                st.metric("Train R²", f"{selected_model['metrics']['train_r2']:.4f}")
            with col4:
                st.metric("Test R²", f"{selected_model['metrics']['test_r2']:.4f}")
            
            st.write(f"**Dataset:** {selected_model['dataset']}")
            st.write(f"**Training Time:** {selected_model['timestamp']}")
        
        with tab2:
            st.subheader("Hyperparameters")
            st.json(selected_model['hyperparameters'])
        
        with tab3:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = selected_model['feature_importance'].head(10)
            sns.barplot(data=top_features, x='Importance', y='Feature', ax=ax)
            ax.set_title('Top 10 Most Important Features')
            st.pyplot(fig)
            
            st.subheader("Feature Importance Table")
            st.dataframe(selected_model['feature_importance'])
        
        with tab4:
            st.subheader("Actual vs Predicted Values")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training set
            ax1.scatter(selected_model['y_train'], selected_model['y_pred_train'], alpha=0.5)
            ax1.plot([selected_model['y_train'].min(), selected_model['y_train'].max()], 
                    [selected_model['y_train'].min(), selected_model['y_train'].max()], 'r--', lw=2)
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title('Training Set')
            
            # Test set
            ax2.scatter(selected_model['y_test'], selected_model['y_pred_test'], alpha=0.5)
            ax2.plot([selected_model['y_test'].min(), selected_model['y_test'].max()], 
                    [selected_model['y_test'].min(), selected_model['y_test'].max()], 'r--', lw=2)
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title('Test Set')
            
            st.pyplot(fig)
    
    # Clear history button
    st.sidebar.header("🗑️ Management")
    if st.sidebar.button("Clear All History", key="clear_history"):
        st.session_state.model_history = []
        st.session_state.current_model = None
        st.rerun()

def visualizations_page():
    """Dedicated visualizations page"""
    st.title("📊 Data Visualizations")
    st.markdown("""
    Explore the dataset through comprehensive visualizations including data distributions 
    and correlation analysis.
    """)
    
    # Load data
    pxp_df, ixi_df = load_data()
    
    if pxp_df is None or ixi_df is None:
        return
    
    # Dataset selection
    st.sidebar.header("📊 Dataset Selection")
    dataset_choice = st.sidebar.radio(
        "Choose dataset:",
        ["PxP (Process x Process)", "IxI (Industry x Industry)"],
        key="viz_dataset_radio",
        help="PxP: Process-level data, IxI: Industry-level data"
    )
    
    selected_data = pxp_df if dataset_choice == "PxP (Process x Process)" else ixi_df
    
    # Display dataset info
    st.subheader("📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Selected Dataset:** {dataset_choice}")
        st.write(f"**Shape:** {selected_data.shape}")
    
    with col2:
        st.write(f"**Number of Features:** {len(selected_data.columns)}")
        st.write(f"**Number of Samples:** {len(selected_data)}")
    
    with col3:
        if st.checkbox("Show dataset info", key="viz_show_info"):
            st.write("**Columns:**")
            for col in selected_data.columns:
                st.write(f"- {col}")
    
    # Preprocess data
    data_log = preprocess_data(selected_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Data Distributions", "🔗 Correlation Matrix", "📋 Raw Data"])
    
    with tab1:
        st.subheader("📊 Log-Transformed Data Distributions")
        
        columns_to_transform = [
            'Value Added [M.EUR]', 
            'Employment [1000 p.]', 
            'GHG emissions [kg CO2 eq.]', 
            'Energy Carrier Net Total [TJ]'
        ]
        
        # Check which columns exist in the data
        existing_columns = [col for col in columns_to_transform if col in data_log.columns]
        
        if not existing_columns:
            st.warning("No transformable columns found in the dataset.")
        else:
            # Create distribution plots
            n_cols = 2
            n_rows = (len(existing_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, column in enumerate(existing_columns):
                if i < len(axes):
                    sns.histplot(data_log[column], kde=True, ax=axes[i])
                    axes[i].set_title(f'Log-Transformed {column}')
                    axes[i].set_xlabel(column)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(existing_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show statistics
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(data_log[existing_columns].describe())
    
    with tab2:
        st.subheader("🔗 Correlation Matrix")
        
        columns_to_transform = [
            'Value Added [M.EUR]', 
            'Employment [1000 p.]', 
            'GHG emissions [kg CO2 eq.]', 
            'Energy Carrier Net Total [TJ]'
        ]
        
        # Filter only columns that exist in the data
        existing_columns = [col for col in columns_to_transform if col in data_log.columns]
        
        if len(existing_columns) < 2:
            st.warning("Need at least 2 columns to create a correlation matrix.")
        else:
            correlation_matrix = data_log[existing_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, ax=ax)
            ax.set_title('Correlation Matrix of Log-Transformed Data')
            st.pyplot(fig)
            
            # Correlation insights
            st.subheader("🔍 Correlation Insights")
            
            # Find strongest correlations
            corr_matrix = correlation_matrix.copy()
            np.fill_diagonal(corr_matrix.values, 0)  # Remove diagonal
            
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Threshold for strong correlation
                        strong_correlations.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_value
                        })
            
            if strong_correlations:
                st.write("**Strong Correlations (|r| > 0.5):**")
                strong_corr_df = pd.DataFrame(strong_correlations)
                st.dataframe(strong_corr_df.sort_values('Correlation', key=abs, ascending=False))
            else:
                st.info("No strong correlations (|r| > 0.5) found between variables.")
    
    with tab3:
        st.subheader("📋 Raw Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_raw = st.checkbox("Show raw data", key="viz_show_raw")
            if show_raw:
                st.dataframe(selected_data)
        
        with col2:
            show_log = st.checkbox("Show log-transformed data", key="viz_show_log")
            if show_log:
                st.dataframe(data_log)
        
        # Data information
        st.subheader("📋 Data Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Data Types:**")
            dtype_info = selected_data.dtypes.reset_index()
            dtype_info.columns = ['Column', 'Data Type']
            st.dataframe(dtype_info, hide_index=True)
        
        with info_col2:
            st.write("**Missing Values:**")
            missing_info = selected_data.isnull().sum().reset_index()
            missing_info.columns = ['Column', 'Missing Values']
            missing_info = missing_info[missing_info['Missing Values'] > 0]
            if len(missing_info) > 0:
                st.dataframe(missing_info, hide_index=True)
            else:
                st.success("No missing values found in the dataset!")

# Main app logic
def main():
    # Display navbar
    create_navbar()
    
    # Page routing based on navbar selection
    if st.session_state.current_page == "main":
        main_page()
    elif st.session_state.current_page == "history":
        history_page()
    elif st.session_state.current_page == "visualizations":
        visualizations_page()

if __name__ == "__main__":
    main()