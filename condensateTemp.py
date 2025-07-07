import pandas as pd
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import numpy as np

st.set_page_config(page_title="Condensate Temperature Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #54565B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold; 
        color: #54565B;
        margin: 1rem 0;
        border-bottom: 2px solid #C5203F;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #C5203F;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Condensate Temperature Dashboard</div>', unsafe_allow_html=True)

csv_dir = ''
CSV_FILE = 'cleaned_csvs/CondensateTemp.csv'
EXCLUDE_COLS = ["Year", "Month", "Week", "Day", "Time", "Date"]

@st.cache_data
def load_data():
    file_path = os.path.join(csv_dir, CSV_FILE)
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        return pd.DataFrame()  # Return empty DataFrame to avoid crashing
    df = pd.read_csv(file_path)
    return df.drop_duplicates()

@st.cache_data
def load_regression_data():
    reg_path = os.path.join(csv_dir, "sum_temp.csv")
    reg_df = pd.read_csv(reg_path)
    reg_df["Date"] = pd.to_datetime(reg_df["Date"])
    reg_df["Year"] = reg_df["Date"].dt.year.astype(str)
    reg_df["Month"] = reg_df["Date"].dt.strftime("%b")
    return reg_df

def get_condensate_meter_columns(df):
    exclude_cols = ["Year", "Month", "Week", "Day", "Time", "Date"]
    return [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

def get_temp_meter_columns(df):
    additional_excludes = ["HDD 15.5", "HDD15.5", "HDD", "10T temp"]
    all_excludes = EXCLUDE_COLS + additional_excludes
    return [col for col in df.columns if col not in all_excludes and pd.api.types.is_numeric_dtype(df[col])]

def calculate_advanced_metrics(df, meter_cols):
    """Calculate advanced performance metrics"""
    metrics = {}
    
    if not meter_cols or df.empty:
        return metrics
    
    # Calculate efficiency metrics
    total_usage = df[meter_cols].sum().sum()
    avg_usage = df[meter_cols].mean().mean()
    peak_usage = df[meter_cols].max().max()
    
    # Load factor (average / peak)
    load_factor = (avg_usage / peak_usage * 100) if peak_usage > 0 else 0
    
    # Variability coefficient
    std_usage = df[meter_cols].std().mean()
    variability = (std_usage / avg_usage * 100) if avg_usage > 0 else 0
    
    metrics.update({
        'load_factor': load_factor,
        'variability': variability,
        'total_usage': total_usage,
        'avg_usage': avg_usage,
        'peak_usage': peak_usage
    })
    
    return metrics

def create_correlation_matrix(df, meter_cols):
    """Create correlation matrix heatmap"""
    if len(meter_cols) < 2:
        return None
    
    corr_matrix = df[meter_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Temperature Meter Correlation Matrix"
    )
    fig.update_layout(
        title_x=0.5,
        height=500,
        template="plotly_white"
    )
    return fig

def create_heatmap_calendar(df, meter_cols):
    """Create calendar heatmap of daily usage"""
    if df.empty or not meter_cols:
        return None
    
    df_daily = df.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['DayOfWeek'] = df_daily['Date'].dt.day_name()
    df_daily['Week'] = df_daily['Date'].dt.isocalendar().week
    
    daily_usage = df_daily.groupby(['Date', 'DayOfWeek', 'Week'])[meter_cols].sum().sum(axis=1).reset_index()
    daily_usage.columns = ['Date', 'DayOfWeek', 'Week', 'TotalUsage']
    
    pivot_data = daily_usage.pivot_table(
        values='TotalUsage', 
        index='DayOfWeek', 
        columns='Week', 
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Week", y="Day of Week", color="Temperature Usage"),
        color_continuous_scale="Viridis",
        title="Daily Temperature Usage Pattern (Calendar Heatmap)"
    )
    fig.update_layout(
        title_x=0.5,
        height=400,
        template="plotly_white"
    )
    return fig

def create_box_plot_analysis(df, meter_cols):
    """Create box plots for usage distribution analysis"""
    if df.empty or not meter_cols:
        return None
    
    # Melt the data for box plot
    df_melted = df[meter_cols + ['Month']].melt(
        id_vars=['Month'],
        value_vars=meter_cols,
        var_name='Temperature_Meter',
        value_name='Temperature'
    )
    
    fig = px.box(
        df_melted,
        x='Month',
        y='Temperature',
        color='Temperature_Meter',
        title="Temperature Distribution by Month"
    )
    fig.update_layout(
        title_x=0.5,
        template="plotly_white",
        height=500
    )
    return fig

def apply_filters(df, date_range=None, selected_years=None, selected_months=None, selected_meters=None):
    """Apply consistent filters to any dataframe"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        try:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df_dates = pd.to_datetime(filtered["Date"], errors='coerce')
            
            # Filter by date range
            date_mask = (df_dates >= start_date) & (df_dates <= end_date)
            filtered = filtered[date_mask]
        except Exception as e:
            st.warning(f"Date filtering error: {e}")
    
    # Apply year filter
    if selected_years and "All" not in selected_years and "Year" in filtered.columns:
        if not filtered.empty:
            # Convert years to strings for consistent comparison
            filtered_years = filtered['Year'].astype(str)
            selected_years_str = [str(y) for y in selected_years]
            filtered = filtered[filtered_years.isin(selected_years_str)]
    
    # Apply month filter
    if selected_months and "All" not in selected_months and "Month" in filtered.columns:
        if not filtered.empty:
            filtered = filtered[filtered['Month'].isin(selected_months)]
    
    # Apply meter filter (keep essential columns but only analyze selected meters)
    if selected_meters and "All" not in selected_meters:
        available_meters = get_temp_meter_columns(filtered) if not filtered.empty else []
        selected_meter_cols = [col for col in selected_meters if col in available_meters]
        
        # Keep essential columns plus selected meters
        essential_cols = ["Year", "Month", "Week", "Day", "Time", "Date"]
        cols_to_keep = []
        
        # Add essential columns that exist
        for col in essential_cols:
            if col in filtered.columns:
                cols_to_keep.append(col)
        
        # Add selected meter columns
        cols_to_keep.extend(selected_meter_cols)
        
        # Always keep HDD 15.5 for regression analysis (but not for plotting)
        if "HDD 15.5" in filtered.columns:
            cols_to_keep.append("HDD 15.5")
        
        # Only filter columns if we have valid selections
        if selected_meter_cols:
            filtered = filtered[cols_to_keep]
    
    return filtered

def filter_by_sidebar(df):
    st.sidebar.markdown("## Dashboard Filters")
    
    # Initialize filters with error handling
    try:
        # Date range filter
        min_date = pd.to_datetime(df["Date"]).min()
        max_date = pd.to_datetime(df["Date"]).max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="sidebar_date_range"
        )
    except Exception:
        date_range = None
        st.sidebar.warning("Date range not available")
    
    # Filter options with validation
    try:
        unique_years = ["All"] + sorted([str(y) for y in df['Year'].unique() if pd.notna(y)])
        unique_months = ["All"] + sorted([m for m in df['Month'].unique() if pd.notna(m)])
        temp_meter_options = ["All"] + get_temp_meter_columns(df)
    except Exception as e:
        st.sidebar.error(f"Error loading filter options: {e}")
        unique_years = ["All"]
        unique_months = ["All"] 
        temp_meter_options = ["All"]

    # Sidebar selections
    selected_meters = st.sidebar.multiselect(
        "Temperature Meters", 
        options=temp_meter_options, 
        default=["All"],
        help="Select specific temperature meters to analyze"
    )
    selected_years = st.sidebar.multiselect(
        "Years", 
        options=unique_years, 
        default=["All"],
        help="Filter by specific years"
    )
    selected_months = st.sidebar.multiselect(
        "Months", 
        options=unique_months, 
        default=["All"],
        help="Filter by specific months"
    )
    
    # Apply filters using the new function
    filtered = apply_filters(df, date_range, selected_years, selected_months, selected_meters)
    
    # Show filter results
    if not filtered.empty:
        st.sidebar.success(f"{len(filtered):,} records after filtering")
    else:
        st.sidebar.error("⚠️ No data matches current filters")
    
    return filtered, selected_meters, selected_years, selected_months, date_range

def main():
    df = load_data()
    filtered, selected_meters, selected_years, selected_months, date_range = filter_by_sidebar(df)
    temp_meter_columns = get_temp_meter_columns(filtered)
    
    # If specific meters are selected, use only those; otherwise use all available temp meters
    if selected_meters and "All" not in selected_meters:
        # Only include meters that are actually in the filtered data and were selected
        metric_cols = [col for col in selected_meters if col in temp_meter_columns]
    else:
        # Use all available temp meter columns (excluding HDD and 10T temp)
        metric_cols = temp_meter_columns

    # ========== SECTION 1: KEY PERFORMANCE INDICATORS ========== 
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate metrics
    highest_recorded = lowest_recorded = highest_avg_temp = "N/A"
    highest_recorded_meter = lowest_recorded_meter = highest_avg_meter = "N/A"
    peak_time = trough_time = "N/A"
    peak_times_list = trough_times_list = []
    
    if not filtered.empty and metric_cols:
        for col in metric_cols:
            filtered[col] = pd.to_numeric(filtered[col], errors='coerce')
        
        # Find highest and lowest recorded temperatures and their meters
        highest_recorded = -999
        lowest_recorded = 999
        for meter in metric_cols:
            meter_data = filtered[meter].dropna()
            if len(meter_data) > 0:
                meter_max = meter_data.max()
                meter_min = meter_data.min()
                if meter_max > highest_recorded:
                    highest_recorded = meter_max
                    highest_recorded_meter = meter
                if meter_min < lowest_recorded:
                    lowest_recorded = meter_min
                    lowest_recorded_meter = meter
        
        # Calculate average temperatures per meter
        meter_avgs = filtered[metric_cols].mean(numeric_only=True)
        highest_avg_temp = meter_avgs.max()
        highest_avg_meter = meter_avgs.idxmax()
        
        # Time-based analysis for peak/trough periods using quartiles
        if "Time" in filtered.columns:
            filtered_time = filtered.copy()
            filtered_time["Hour"] = pd.to_datetime(filtered_time["Time"], errors="coerce").dt.hour
            hourly_avg = filtered_time.groupby("Hour")[metric_cols].mean().mean(axis=1)
            if not hourly_avg.empty:
                # Calculate quartiles
                q1 = hourly_avg.quantile(0.10)
                q3 = hourly_avg.quantile(0.90)
                
                # Peak times: hours in the top quartile (Q3 and above)
                peak_hours = hourly_avg[hourly_avg >= q3].index.tolist()
                peak_times_list = [f"{hour:02d}:00" for hour in sorted(peak_hours)]
                peak_time = peak_times_list[0] if peak_times_list else "N/A"
                
                # Trough times: hours in the bottom quartile (Q1 and below)
                trough_hours = hourly_avg[hourly_avg <= q1].index.tolist()
                trough_times_list = [f"{hour:02d}:00" for hour in sorted(trough_hours)]
                trough_time = trough_times_list[0] if trough_times_list else "N/A"

    with col1:
        st.metric(
            "Highest Recorded (°C)", 
            f"{highest_recorded:.2f}" if isinstance(highest_recorded, (int, float)) and highest_recorded != -999 else "N/A",
            border=True
        )
        if highest_recorded_meter != "N/A":
            st.selectbox("Meter:", [highest_recorded_meter], key="highest_meter", disabled=True)
    
    with col2:
        st.metric(
            "Lowest Recorded (°C)", 
            f"{lowest_recorded:.2f}" if isinstance(lowest_recorded, (int, float)) and lowest_recorded != 999 else "N/A",
            border=True
        )
        if lowest_recorded_meter != "N/A":
            st.selectbox("Meter:", [lowest_recorded_meter], key="lowest_meter", disabled=True)
    
    with col3:
        st.metric(
            "Highest Avg Temp (°C)", 
            f"{highest_avg_temp:.2f}" if isinstance(highest_avg_temp, (int, float)) else "N/A",
            border=True
        )
        if highest_avg_meter != "N/A":
            st.selectbox("Meter:", [highest_avg_meter], key="highest_avg_meter", disabled=True)
    
    with col4:
        st.metric(
            "Peak Time", 
            peak_time,
            border=True
        )
        if peak_times_list:
            st.selectbox("All Peak Times:", peak_times_list, index=0, key="peak_times")
    
    with col5:
        st.metric(
            "Trough Time", 
            trough_time,
            border=True
        )
        if trough_times_list:
            st.selectbox("All Trough Times:", trough_times_list, index=0, key="trough_times")

    # ========== SECTION 2: TIME SERIES ANALYSIS ==========
    st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    col_ts1, col_ts2 = st.columns([3, 1])
    
    with col_ts2:
        resample_interval = st.selectbox(
            "Time Interval",
            options=[("15 Minutes", "15T"), ("1 Hour", "H"), ("1 Day", "D")],
            format_func=lambda x: x[0],
            index=1
        )[1]
        
        chart_type = st.selectbox(
            "Chart Type",
            options=["Line", "Area"],
            index=0
        )
    
    with col_ts1:
        if not filtered.empty and metric_cols:
            filtered_ts = filtered.copy()
            if "Date" in filtered_ts.columns and "Time" in filtered_ts.columns:
                filtered_ts["DateTime"] = pd.to_datetime(
                    filtered_ts["Date"].astype(str) + " " + filtered_ts["Time"].astype(str), 
                    errors="coerce"
                )
                filtered_ts = filtered_ts.dropna(subset=["DateTime"])
                filtered_ts = filtered_ts.set_index("DateTime").resample(resample_interval).mean(numeric_only=True).reset_index()
                x_col = "DateTime"
            else:
                x_col = "Time" if "Time" in filtered_ts.columns else filtered_ts.columns[0]
            
            fig = go.Figure()
            meters_to_plot = metric_cols if metric_cols else temp_meter_columns
            for meter in meters_to_plot:
                if meter in filtered_ts.columns:
                    if chart_type == "Area":
                        fig.add_trace(go.Scatter(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines',
                            fill='tonexty' if meter != meters_to_plot[0] else 'tozeroy',
                            name=meter,
                            stackgroup='one'
                        ))
                    else:  # Line chart
                        fig.add_trace(go.Scattergl(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines+markers',
                            name=meter,
                            hovertemplate=f"<b>{meter}</b><br>Temperature: %{{y}}°C<extra></extra>"
                        ))
            
            fig.update_layout(
                title="Temperature Over Time",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                height=500,
                xaxis=dict(
                    showgrid=True,
                    rangeslider=dict(visible=True) if chart_type == "Line" else dict(visible=False)
                ),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

# ========== DETAILED METER METRICS ==========
    st.markdown('<div class="section-header">Detailed Meter Analysis</div>', unsafe_allow_html=True)
    
    if not filtered.empty and metric_cols:
        # Create detailed metrics table
        metrics_data = []
        for meter in metric_cols:
            meter_data = filtered[meter].dropna()
            if len(meter_data) > 0:
                metrics_data.append({
                    'Meter': meter,
                    'Avg Temp (°C)': round(meter_data.mean(), 2),
                    'Max Temp (°C)': round(meter_data.max(), 2),
                    'Min Temp (°C)': round(meter_data.min(), 2),
                    'Range (°C)': round(meter_data.max() - meter_data.min(), 2),
                    'Std Dev (°C)': round(meter_data.std(), 2),
                    'Readings Count': len(meter_data)
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Time analysis chart
            col_time1, col_time2 = st.columns(2)
            
            with col_time1:
                if "Time" in filtered.columns:
                    filtered_time = filtered.copy()
                    filtered_time["Hour"] = pd.to_datetime(filtered_time["Time"], errors="coerce").dt.hour
                    hourly_avg = filtered_time.groupby("Hour")[metric_cols].mean().mean(axis=1).reset_index()
                    hourly_avg.columns = ["Hour", "Avg Temperature"]
                    
                    fig_hourly = px.line(
                        hourly_avg,
                        x="Hour",
                        y="Avg Temperature",
                        title="Average Temperature by Hour of Day",
                        markers=True
                    )
                    fig_hourly.update_layout(height=300, template="plotly_white")
                    st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col_time2:
                # Temperature range comparison
                range_data = [(row['Meter'], row['Range (°C)']) for row in metrics_data]
                range_df = pd.DataFrame(range_data, columns=['Meter', 'Temperature Range'])
                
                fig_range = px.bar(
                    range_df,
                    x="Temperature Range",
                    y="Meter",
                    orientation='h',
                    title="Temperature Range by Meter",
                    color="Temperature Range",
                    color_continuous_scale="Viridis"
                )
                fig_range.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig_range, use_container_width=True)
    else:
        st.info("No data available for detailed metrics analysis.")

    # ========== SECTION 3: ADVANCED ANALYTICS ========== 
    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Temperature Patterns", "Correlations", "Calendar View", "Distributions"])
    
    with tab1:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if not filtered.empty and metric_cols:
                avg_temps = filtered[metric_cols].mean().reset_index()
                avg_temps.columns = ["Meter", "Avg Temperature"]
                avg_temps = avg_temps[avg_temps["Avg Temperature"].notna()]
                if not avg_temps.empty:
                    fig_pie = px.pie(
                        avg_temps,
                        names="Meter",
                        values="Avg Temperature",
                        title="Average Temperature Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
        with col_p2:
            if not filtered.empty and metric_cols:
                filtered_copy = filtered.copy()
                if "DateTime" in filtered_copy.columns:
                    filtered_copy["Hour"] = filtered_copy["DateTime"].dt.hour
                elif "Time" in filtered_copy.columns:
                    filtered_copy["Hour"] = pd.to_datetime(filtered_copy["Time"], errors="coerce").dt.hour
                else:
                    filtered_copy["Hour"] = 12
                filtered_copy["Period"] = filtered_copy["Hour"].apply(
                    lambda x: "Day (6AM-6PM)" if 6 <= x < 18 else "Night (6PM-6AM)"
                )
                period_temp = filtered_copy.groupby("Period")[metric_cols].mean().mean(axis=1).reset_index()
                period_temp.columns = ["Period", "Avg Temperature"]
                fig_period = px.bar(
                    period_temp,
                    x="Period",
                    y="Avg Temperature",
                    title="Day vs Night Avg Temperature",
                    color="Period",
                    color_discrete_sequence=["#FFA500", "#4169E1"]
                )
                fig_period.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_period, use_container_width=True)
    with tab2:
        if not filtered.empty and len(metric_cols) > 1:
            corr_fig = create_correlation_matrix(filtered, metric_cols)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Need at least 2 meters for correlation analysis.")
    with tab3:
        if not filtered.empty and metric_cols:
            calendar_fig = create_heatmap_calendar(filtered, metric_cols)
            if calendar_fig:
                st.plotly_chart(calendar_fig, use_container_width=True)
        else:
            st.info("No data available for calendar view.")
    with tab4:
        if not filtered.empty and metric_cols:
            box_fig = create_box_plot_analysis(filtered, metric_cols)
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("No data available for distribution analysis.")

if __name__ == "__main__":
    main()
