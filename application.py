# Import packages
from dash import Dash, html, dcc, Output, Input, State
import dash_auth
import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors
from dash.exceptions import PreventUpdate
import time
from sklearn.utils import resample
import dask.dataframe as dd
from dask.distributed import Client
import multiprocessing as mp

# Defining some global variables for easy adaptability
title = 'COMBINED ATLAS'
reduction = 'UMAP'
id_meta = f'{reduction} with metadata'
id_counts = f'{reduction} with count data'
count1 = 'raw_counts'
count2 = 'imputed_counts'
covariate1 = 'batch'
covariate2 = 'sample'
covariate3 = 'stage'
meta1 = 'query_cat'
meta2 = 'celltype_extended_atlas'
meta3 = 'celltype_nowotschin'
meta4 = 'cell'
meta_of_interest = [covariate1, covariate3, meta2, meta3]
default_gene = 'Sox17'
default_meta = 'batch'
default_counts = 'raw_counts'
covariates_composite = 'covariates_composite'
sampled_quantity = int(35000)


# Load data
def load_data():
    # Setting dtypes of meta
    meta_dtypes = {
        "cell": "object",
        "batch": "category",
        "sample": "category",
        "stage": "category",
        "celltype_extended_atlas": "category",
        "celltype_nowotschin": "category",
        "query": "category",
        "query_cat": "category",
        "colormap_batch": "category",
        "symbolmap_plotly_batch": "category",
        "colormap_stage": "category",
        "symbolmap_plotly_stage": "category",
        "colormap_celltype_extended_atlas": "category",
        "colormap_celltype_nowotschin": "category",
        "UMAP1": np.float64,
        "UMAP2": np.float64,
        "covariates_composite": "category"
    }
    # Define the data path
    app_data = '/mnt/efs/'
    # Load the meta_table from a CSV file
    meta_table_path = os.path.join(app_data, 'meta_table.csv')
    meta_table = pd.read_csv(meta_table_path, index_col=0, dtype=meta_dtypes)

    # Use Dask to lazily load raw_counts and imputed_counts
    raw_counts_path = os.path.join(app_data, f'{count1}.parquet')
    imputed_counts_path = os.path.join(app_data, f'{count2}.parquet')
    raw_counts = dd.read_parquet(raw_counts_path)
    imputed_counts = dd.read_parquet(imputed_counts_path)

    # Define symbol map
    symbolmap = {'ref': 'circle', 'query': 'star'}
    meta_table['markersymbol'] = meta_table[meta1].map(symbolmap)

    # Marker size
    sizemap = {'ref': 5, 'query': 10}
    meta_table['markersize'] = meta_table[meta1].map(sizemap)

    column_names = raw_counts.columns.tolist()

    return meta_table, raw_counts, imputed_counts, column_names

def stratified_sample(data, by_column, total_samples):
    sampled_meta = []

    # Calculate the value counts and filter out zero counts
    value_counts = data[by_column].value_counts()
    value_counts = value_counts[value_counts > 0]  # Filter out zero counts

    # Calculate the log2 of the proportions
    covariate_composite_log_counts = np.log2(value_counts)
    proportions_log = covariate_composite_log_counts / sum(covariate_composite_log_counts)

    # Iterate over each class in by_column
    for class_value, proportion in proportions_log.items():
        class_data = data[data[by_column] == class_value]

        # Calculate the number of samples for this class based on the proportion
        n_samples = int(proportion * total_samples)

        # If there are fewer samples than the calculated number, use all samples
        if len(class_data) <= n_samples:
            sampled_meta.append(class_data)
        else:
            ## avoiding zero samples again
            n_samples=max(1, n_samples)
            # Resample the data to the calculated number of samples
            sampled_class_data = resample(class_data, replace=False, n_samples=n_samples, random_state=42)
            sampled_meta.append(sampled_class_data)

    # Concatenate all the sampled data
    sampled_meta = pd.concat(sampled_meta)
    return sampled_meta


    # Concatenate all the sampled data
    sampled_meta = pd.concat(sampled_meta)

    return sampled_meta

# Convert Matplotlib colormap to Plotly colorscale
def matplotlib_to_plotly(cmap, pl_entries=255):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, f'rgb{tuple(C)}'])
    return pl_colorscale

# Utility function to handle zoom data extraction
def get_zoom_limits(zoom_data, default_x_min, default_x_max, default_y_min, default_y_max):
    x_min = zoom_data.get('x_min', default_x_min)
    x_max = zoom_data.get('x_max', default_x_max)
    y_min = zoom_data.get('y_min', default_y_min)
    y_max = zoom_data.get('y_max', default_y_max)
    return x_min, x_max, y_min, y_max

# Utility function to create background trace (common for both meta and counts)
def create_background_trace(x, y, visibility):
    return go.Scattergl(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=3, color='lightgray', opacity=0.2),
        name='All data',
        hoverinfo='skip',
        showlegend=False,
        visible=visibility
    )

def create_dash_app(meta_table, raw_counts, imputed_counts, column_names):
    app = Dash(__name__)
    ## setting username and password
    username_password = {'some_username': 'some_password'}
    auth = dash_auth.BasicAuth(app, username_password)

    # Define the color map
    colors = [(0.9, 0.9, 0.9), (0.7, 0.7, 0.7), (0.5, 0.5, 1.0), (0.0, 0.0, 0.2)]  # very light gray to very dark blue
    n_bins = 100
    cmap_name = 'light_gray_to_dark_blue'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    colormap_plotly = matplotlib_to_plotly(custom_cmap)

    # App layout
    app.layout = html.Div([
        html.H1(title),
        html.Hr(),
        dcc.Dropdown(
            id='dropdown_meta',
            options=[{'label': meta, 'value': meta} for meta in meta_of_interest],
            value=default_meta
        ),
        dcc.Graph(id=id_meta),
        dcc.Store(id='zoom_data_meta', data={}),  # Store zoom information for meta plot
        html.Hr(),
        html.Label(f'Gene: '),
        dcc.Dropdown(
            id='dropdown_gene',
            options=[{'label': name, 'value': name} for name in column_names],
            value=default_gene  # Default value
        ),
        html.Label('Count type:'),
        dcc.Dropdown(
            id='dropdown_counts',
            options=[
                {'label': 'Raw Counts', 'value': 'raw_counts'},
                {'label': 'Imputed Counts', 'value': 'imputed_counts'}
            ],
            value=default_counts  # Default value
        ),
        dcc.Graph(id=id_counts),
        dcc.Store(id='zoom_data_counts', data={})  # Store zoom information for expression plot
    ])

    @app.callback(
        Output(id_meta, 'figure'),
        [Input('dropdown_meta', 'value'), 
         Input('zoom_data_meta', 'data')]
    )
    def update_graph_meta(selected_column, zoom_data):
        ## start timing
        start_time = time.time()
        ## if selected column does not exist
        if selected_column is None:
            raise PreventUpdate

        # Get the zoom limits
        x_min, x_max, y_min, y_max = get_zoom_limits(
            zoom_data,
            meta_table[f'{reduction}1'].min(),
            meta_table[f'{reduction}1'].max(),
            meta_table[f'{reduction}2'].min(),
            meta_table[f'{reduction}2'].max()
        )

        # Filter data based on zoom limits
        filtered_data = meta_table[
            (meta_table[f'{reduction}1'] >= x_min) &
            (meta_table[f'{reduction}1'] <= x_max) &
            (meta_table[f'{reduction}2'] >= y_min) &
            (meta_table[f'{reduction}2'] <= y_max)
        ]

        # Resample the data
        sampled_meta = stratified_sample(filtered_data, covariates_composite, sampled_quantity)
        unique_values = sampled_meta[selected_column].unique()

        # Sort unique values alphanumerically
        unique_values = sorted(unique_values, key=lambda x: (str(x).isdigit(), str(x)))


        # Create the background trace
        traces = [create_background_trace(meta_table[f'{reduction}1'], meta_table[f'{reduction}2'], True)]

        # Add the foreground traces
        for value in unique_values:
            value_data = sampled_meta[sampled_meta[selected_column] == value]
            trace = go.Scattergl(
                x=value_data[f'{reduction}1'],
                y=value_data[f'{reduction}2'],
                mode='markers',
                marker=dict(
                    size=value_data['markersize'],
                    symbol=value_data['markersymbol'],
                    color=value_data[f'colormap_{selected_column}']
                ),
                name=str(value),
                hoverinfo='text',
                hovertext = value_data.apply(
                    lambda row: (
                        f"{meta3}: {row[meta3]}<br>"
                        f"{meta2}: {row[meta2]}<br>"
                        "-<br>"
                        f"{covariate3}: {row[covariate3]}<br>"
                        f"{covariate1}: {row[covariate1]}<br>"
                        f"{covariate2}: {row[covariate2]}"
                    ),
                    axis=1
                )
            )
            traces.append(trace)

        # Create and return the updated figure
        fig = go.Figure(data=traces)
        # set some visual parameters
        fig.update_layout(
            height=750,
            width=1250,
            title=f'{id_meta} colored by: {selected_column}',
            xaxis_title=f'{reduction}1',
            yaxis_title=f'{reduction}2',
            xaxis=dict(showgrid=False, showticklabels=False, range=[x_min, x_max] if 'x_min' in zoom_data else None),
            yaxis=dict(showgrid=False, showticklabels=False, range=[y_min, y_max] if 'y_min' in zoom_data else None),
            plot_bgcolor='white'
        )
        print(f"Metadata plot time: {time.time() - start_time} seconds")
        return fig

    @app.callback(
        Output(id_counts, 'figure'),
        [Input('dropdown_gene', 'value'), Input('dropdown_counts', 'value'), Input('zoom_data_counts', 'data')]
    )
    def update_graph_counts(selected_column, color_by, zoom_data):
        start_time = time.time()
        if selected_column is None or color_by is None:
            raise PreventUpdate

        x_min, x_max, y_min, y_max = get_zoom_limits(
            zoom_data,
            meta_table[f'{reduction}1'].min(),
            meta_table[f'{reduction}1'].max(),
            meta_table[f'{reduction}2'].min(),
            meta_table[f'{reduction}2'].max()
        )

        filtered_data = meta_table[
            (meta_table[f'{reduction}1'] >= x_min) &
            (meta_table[f'{reduction}1'] <= x_max) &
            (meta_table[f'{reduction}2'] >= y_min) &
            (meta_table[f'{reduction}2'] <= y_max)
        ]

        sampled_meta = stratified_sample(filtered_data, covariates_composite, sampled_quantity)

        # reading in count information
        raw_counts_selected = raw_counts[selected_column].compute()
        imputed_counts_selected = imputed_counts[selected_column].compute()
        count_type_selected = raw_counts_selected if color_by == 'raw_counts' else imputed_counts_selected

        sampled_log_counts = np.log2(count_type_selected.loc[sampled_meta.index] + 1)

        # checking unique stages to add as trace
        unique_stages = sampled_meta[covariate3].unique()
        # Sort unique values alphanumerically
        unique_stages = sorted(unique_stages, key=lambda x: (str(x).isdigit(), str(x)))

        # adding the invariable background trace
        traces = [create_background_trace(meta_table[f'{reduction}1'], meta_table[f'{reduction}2'], True)]

        # adding stage specific traces
        for stage in unique_stages:
            stage_data = sampled_meta[sampled_meta[covariate3] == stage]
            trace = go.Scattergl(
                x=stage_data[f'{reduction}1'],
                y=stage_data[f'{reduction}2'],
                mode='markers',
                marker=dict(
                    color=sampled_log_counts.loc[stage_data.index],
                    colorscale=colormap_plotly,
                    coloraxis='coloraxis',
                    size=stage_data['markersize'],
                    symbol=stage_data['markersymbol']
                ),
                name=str(stage),
                hoverinfo='text',
                hovertext = stage_data.apply(
                    lambda row: (
                        f"{count1}: {raw_counts_selected.loc[row.name]}<br>"
                        f"{count2}: {imputed_counts_selected.loc[row.name]:.3f}<br>"
                        "-<br>"
                        f"{meta3}: {row[meta3]}<br>"
                        f"{meta2}: {row[meta2]}<br>"
                        "-<br>"
                        f"{covariate3}: {row[covariate3]}<br>"
                        f"{covariate1}: {row[covariate1]}<br>"
                        f"{covariate2}: {row[covariate2]}"
                    ), 
                    axis=1
                )
            )
            traces.append(trace)
        ## assembling a figure from all the traces
        fig = go.Figure(data=traces)
        ## setting some visual parameters
        fig.update_layout(
            title=f'{reduction} colored by {color_by}: {selected_column}',
            xaxis_title=f'{reduction}1',
            yaxis_title=f'{reduction}2',
            legend=dict(
                orientation="v",
                x=1.12,  # Positioning the legend next to the colorbar
                y=.95,
                yanchor="top"
            ),
            height=750,
            width=1250,
            coloraxis=dict(
                colorscale=colormap_plotly,
                colorbar=dict(title='log2(counts + 1)')
            ),
            xaxis=dict(showgrid=False, showticklabels=False, range=[x_min, x_max] if 'x_min' in zoom_data else None),
            yaxis=dict(showgrid=False, showticklabels=False, range=[y_min, y_max] if 'y_min' in zoom_data else None),
            plot_bgcolor='white'
        )
        ## returning figure
        print(f"Gene expression plot time: {time.time() - start_time} seconds")
        return fig

    @app.callback(
        Output('zoom_data_meta', 'data'),
        Input(id_meta, 'relayoutData')
    )
    def store_zoom_data_meta(relayout_data):
        if relayout_data is None:
            raise PreventUpdate

        zoom_data = {}
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            zoom_data['x_min'] = relayout_data['xaxis.range[0]']
            zoom_data['x_max'] = relayout_data['xaxis.range[1]']
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            zoom_data['y_min'] = relayout_data['yaxis.range[0]']
            zoom_data['y_max'] = relayout_data['yaxis.range[1]']

        return zoom_data

    @app.callback(
        Output('zoom_data_counts', 'data'),
        Input(id_counts, 'relayoutData')
    )
    def store_zoom_data_expression(relayout_data):
        if relayout_data is None:
            raise PreventUpdate

        zoom_data = {}
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            zoom_data['x_min'] = relayout_data['xaxis.range[0]']
            zoom_data['x_max'] = relayout_data['xaxis.range[1]']
        if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
            zoom_data['y_min'] = relayout_data['yaxis.range[0]']
            zoom_data['y_max'] = relayout_data['yaxis.range[1]']

        return zoom_data

    return app

# Load data and create app in any case
meta_table, raw_counts, imputed_counts, column_names = load_data()
app = create_dash_app(meta_table, raw_counts, imputed_counts, column_names)

# Run the app if executed locally as a script
if __name__ == '__main__':
    mp.set_start_method('spawn')
    client = Client(n_workers=2, threads_per_worker=5)
    
    # Create and run the Dash app
    app.run_server(debug=False, host='0.0.0.0', port=8000)

# Run the app if employed as a module (e.g., gunicorn)
else: 
    # Set a secret key for Flask and gunicorn
    app.server.secret_key = 'some_password'
    server = app.server
