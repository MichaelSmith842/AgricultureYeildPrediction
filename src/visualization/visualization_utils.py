#!/usr/bin/env python3
"""
Visualization utilities for the Agricultural Yield Climate Impact Analysis System.
Provides functions to create various plots and visualizations of agricultural and climate data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def set_plotting_style():
    """Set consistent style for matplotlib visualizations."""
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 12

def create_time_series_plot(df, x_col, y_col, color_col=None, title=None, x_label=None, y_label=None):
    """
    Create a time series plot using Plotly Express.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis (typically time)
        y_col (str): Column name for y-axis
        color_col (str, optional): Column name for color differentiation
        title (str, optional): Plot title
        x_label (str, optional): X-axis label
        y_label (str, optional): Y-axis label
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set defaults for labels if not provided
    if title is None:
        title = f"{y_col} over {x_col}"
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    # Create the plot
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title=title,
        labels={x_col: x_label, y_col: y_label}
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text=color_col if color_col else "",
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    return fig

def create_bar_chart(df, x_col, y_col, color_col=None, title=None, x_label=None, y_label=None, barmode="group"):
    """
    Create a bar chart using Plotly Express.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        color_col (str, optional): Column name for color differentiation
        title (str, optional): Plot title
        x_label (str, optional): X-axis label
        y_label (str, optional): Y-axis label
        barmode (str, optional): Bar mode ('group', 'stack', etc.)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set defaults for labels if not provided
    if title is None:
        title = f"{y_col} by {x_col}"
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    # Create the plot
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        barmode=barmode
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text=color_col if color_col else "",
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, size_col=None, title=None, x_label=None, y_label=None, add_trendline=False):
    """
    Create a scatter plot using Plotly Express.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        color_col (str, optional): Column name for color differentiation
        size_col (str, optional): Column name for point size
        title (str, optional): Plot title
        x_label (str, optional): X-axis label
        y_label (str, optional): Y-axis label
        add_trendline (bool, optional): Whether to add a trendline
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set defaults for labels if not provided
    if title is None:
        title = f"{y_col} vs {x_col}"
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    # Check if statsmodels is available for trendline
    trendline_param = None
    if add_trendline:
        try:
            import statsmodels.api as sm
            trendline_param = 'ols'
        except ImportError:
            print("Warning: statsmodels not available. Trendline will not be shown.")
            # Continue without trendline
    
    # Create the plot
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        trendline=trendline_param
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text=color_col if color_col else "",
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    # If we couldn't add a trendline using statsmodels, add a simple linear regression line
    if add_trendline and trendline_param is None:
        # Calculate linear regression manually
        x = df[x_col].values
        y = df[y_col].values
        
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) > 1:  # Need at least 2 points for regression
            # Simple linear regression: y = mx + b
            n = len(x)
            m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
            b = (np.sum(y) - m * np.sum(x)) / n
            
            # Add the trend line
            x_range = np.linspace(min(x), max(x), 100)
            y_range = m * x_range + b
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend'
                )
            )
    
    return fig

def create_heatmap(data, x_col, y_col, value_col, title=None, x_label=None, y_label=None, colorscale="YlOrRd"):
    """
    Create a heatmap using Plotly Express.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        value_col (str): Column name for values (color intensity)
        title (str, optional): Plot title
        x_label (str, optional): X-axis label
        y_label (str, optional): Y-axis label
        colorscale (str, optional): Colorscale for the heatmap
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set defaults for labels if not provided
    if title is None:
        title = f"{value_col} by {x_col} and {y_col}"
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    # Pivot the data if it's not already in the right format
    if isinstance(data, pd.DataFrame) and x_col in data.columns and y_col in data.columns:
        pivot_data = data.pivot_table(index=y_col, columns=x_col, values=value_col)
    else:
        # Assume data is already pivoted
        pivot_data = data
    
    # Create the heatmap
    fig = px.imshow(
        pivot_data,
        labels={"x": x_label, "y": y_label, "color": value_col},
        x=pivot_data.columns,
        y=pivot_data.index,
        title=title,
        color_continuous_scale=colorscale
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    return fig

def create_correlation_matrix(df, columns=None, title="Correlation Matrix"):
    """
    Create a correlation matrix visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        columns (list, optional): List of columns to include in the correlation matrix
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Select columns if specified
    if columns is not None:
        data = df[columns].copy()
    else:
        # Only include numeric columns
        data = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create the heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        labels={"x": "", "y": "", "color": "Correlation"},
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        title=title,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=600,
        width=700
    )
    
    return fig

def create_feature_importance_plot(importance_df, title="Feature Importance"):
    """
    Create a bar chart of feature importance.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with columns 'Feature' and 'Importance'
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Sort by importance
    sorted_df = importance_df.sort_values('Importance', ascending=True).tail(15)
    
    # Create the bar chart
    fig = px.bar(
        sorted_df,
        y='Feature',
        x='Importance',
        title=title,
        orientation='h'
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        yaxis_title="",
        xaxis_title="Importance",
        height=600,
        width=800
    )
    
    return fig

def create_actual_vs_predicted_plot(prediction_df, title="Actual vs Predicted Values"):
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        prediction_df (pd.DataFrame): DataFrame with columns 'Actual' and 'Predicted'
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create the plot
    fig = px.scatter(
        prediction_df,
        x='Actual',
        y='Predicted',
        title=title
    )
    
    # Add identity line (perfect prediction)
    min_val = min(prediction_df['Actual'].min(), prediction_df['Predicted'].min())
    max_val = max(prediction_df['Actual'].max(), prediction_df['Predicted'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=600,
        width=800
    )
    
    return fig

def create_scenario_heatmap(scenario_results, crop, model_type, title=None):
    """
    Create a heatmap visualization of scenario analysis results.
    
    Args:
        scenario_results (pd.DataFrame): DataFrame with scenario results
        crop (str): Crop to filter by
        model_type (str): Model type to filter by
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Filter data
    filtered_data = scenario_results[
        (scenario_results['Crop'] == crop) & 
        (scenario_results['ModelType'] == model_type)
    ]
    
    # Pivot data for heatmap
    pivot_data = filtered_data.pivot_table(
        index='TemperatureChange',
        columns='PrecipitationChange',
        values='PredictedYield'
    )
    
    # Set default title
    if title is None:
        title = f"Predicted {crop} Yield Under Climate Scenarios ({model_type})"
    
    # Create the heatmap
    fig = px.imshow(
        pivot_data,
        labels={"x": "Precipitation Change (%)", 
                "y": "Temperature Change (°C)", 
                "color": "Predicted Yield"},
        x=pivot_data.columns,
        y=pivot_data.index,
        title=title,
        color_continuous_scale="YlGnBu"
    )
    
    # Add text annotations with values
    for i, temp in enumerate(pivot_data.index):
        for j, precip in enumerate(pivot_data.columns):
            value = pivot_data.iloc[i, j]
            fig.add_annotation(
                x=precip,
                y=temp,
                text=f"{value:.1f}",
                showarrow=False,
                font=dict(color="black" if value > pivot_data.mean().mean() else "white")
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=600,
        width=800,
        xaxis_title="Precipitation Change (%)",
        yaxis_title="Temperature Change (°C)"
    )
    
    return fig

def create_multi_crop_comparison(df, year_col, value_col, crop_col, title=None):
    """
    Create a multi-crop comparison visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        year_col (str): Column name for years
        value_col (str): Column name for values to plot
        crop_col (str): Column name for crop types
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Set default title
    if title is None:
        title = f"{value_col} Comparison Across Crops"
    
    # Create subplots (2x4 grid for 8 crops)
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=sorted(df[crop_col].unique()),
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    # Add traces for each crop
    for i, crop in enumerate(sorted(df[crop_col].unique())):
        crop_data = df[df[crop_col] == crop]
        
        # Calculate subplot position
        row = (i // 4) + 1
        col = (i % 4) + 1
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=crop_data[year_col],
                y=crop_data[value_col],
                name=crop
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=False,
        template="plotly_white",
        height=800,
        width=1200
    )
    
    # Update y-axis titles
    for i in range(1, 3):
        fig.update_yaxes(title_text=value_col, row=i, col=1)
    
    # Update x-axis titles
    for i in range(1, 5):
        fig.update_xaxes(title_text=year_col, row=2, col=i)
    
    return fig

def plot_climate_yield_relationship(climate_df, yield_df, climate_var, crop, merged_col='YEAR', title=None):
    """
    Create a dual-axis plot showing climate variable and crop yield over time.
    
    Args:
        climate_df (pd.DataFrame): DataFrame with climate data
        yield_df (pd.DataFrame): DataFrame with yield data
        climate_var (str): Column name of climate variable to plot
        crop (str): Crop type to filter yield data
        merged_col (str): Column to merge dataframes on (typically year)
        title (str, optional): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Filter yield data for the specific crop
    crop_yield = yield_df[yield_df['CROP'] == crop].copy()
    
    # Merge data on the common column
    merged_data = pd.merge(
        crop_yield[['YEAR', 'YIELD']],
        climate_df[[merged_col, climate_var]],
        left_on='YEAR',
        right_on=merged_col,
        how='inner'
    )
    
    # Set default title
    if title is None:
        title = f"{crop} Yield and {climate_var} Relationship Over Time"
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add yield trace
    fig.add_trace(
        go.Scatter(
            x=merged_data['YEAR'],
            y=merged_data['YIELD'],
            name=f"{crop} Yield",
            mode='lines+markers',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add climate variable trace
    fig.add_trace(
        go.Scatter(
            x=merged_data['YEAR'],
            y=merged_data[climate_var],
            name=climate_var,
            mode='lines+markers',
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set axis titles
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text=f"{crop} Yield", secondary_y=False)
    fig.update_yaxes(title_text=climate_var, secondary_y=True)
    
    return fig

if __name__ == "__main__":
    # Test code to verify functionality
    import os
    
    # Sample paths
    sample_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample")
    usda_sample_path = os.path.join(sample_data_dir, "usda_crop_data_sample.csv")
    
    # Load sample data
    try:
        usda_data = pd.read_csv(usda_sample_path)
        
        # Test time series plot
        corn_data = usda_data[usda_data['CROP'] == 'CORN']
        fig = create_time_series_plot(
            corn_data, 
            x_col='YEAR', 
            y_col='YIELD', 
            title='Corn Yield Over Time',
            x_label='Year',
            y_label='Yield (bushels/acre)'
        )
        
        # Save the plot as HTML
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "figures")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "corn_yield_time_series.html"))
        
        print("Test visualization created successfully!")
        
    except FileNotFoundError:
        print("Sample data files not found. Please run the data download scripts first.")