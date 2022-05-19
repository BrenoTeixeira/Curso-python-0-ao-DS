import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import datetime as dt
import plotly.express as px
print("")
sns.set_theme(style='darkgrid', font='Source Sans Pro')
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
# Extract
def get_data(pat):
    data = pd.read_csv(pat, parse_dates=['date'])
    return data


def get_geofile(url):
    geo = geopandas.read_file(url)
    return geo


# Transformation
def set_feature(data):
    data['price_m²'] = data['price']/data['m²_lot']
    return data


def null_values(data):
    data = data.replace(to_replace='no_info', value='')
    return data


### Load ###
def descriptive_stats(data):

    stats = data.describe().drop(['25%', '50%', '75%'], axis=0).T
    return stats


def summary_by_region(data):
    quant = data[['zipcode', 'm²_living', 'price', 'price_m²']].groupby('zipcode').size()
    group = pd.pivot_table(data, values=['m²_living', 'price', 'price_m²'], index='zipcode', aggfunc='mean')
    group['number_of_houses'] = quant
    return group[['number_of_houses', 'm²_living', 'price', 'price_m²']].reset_index()


def data_overview(data):
    ################
    # Data Overview##
    ################
    st.sidebar.header('Data Overview Options')
    f_attributes = st.sidebar.multiselect('Enter Columns', list(data.columns))
    f_zipcode = st.sidebar.multiselect('Enter Region', list(data['zipcode'].unique()))

    st.header('Table Filtered by Region and Attributes')
    if f_zipcode != [] and f_attributes != []:
        df_0 = data.loc[data['zipcode'].isin(f_zipcode)][f_attributes]
    elif f_zipcode != [] and f_attributes == []:
        df_0 = data.loc[data['zipcode'].isin(f_zipcode)]
    elif f_zipcode == [] and f_attributes != []:
        df_0 = data[f_attributes]
    else:
        df_0 = data.copy()

    st.write(df_0.head())
    c1, c2 = st.beta_columns((1, 1))

    c1.header('Summary')
    summ = summary_by_region(df_0)
    c1.dataframe(summ, width=600, height=600)
    c2.header('Descriptive Table')
    stats = descriptive_stats(df_0)
    c2.dataframe(stats, width=600, height=600)
    return None


def portfolio_density(data, geo_file):
    # Base Map Folium
    st.header('Region Overview')
    c, c3 = st.beta_columns((1, 1))

    c.header('Portifolio Density')
    df_1 = data.copy()

    density_map = folium.Map(location=[df_1['lat'].mean(), df_1['long'].mean()], default_zoom=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df_1.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup=f"Sold R${row['price']} on: {row['date']}. "
                            f"Features: {row['m²_living']}. {row['bedrooms']} "
                            f"bedrooms. {row['bathrooms']} bathrooms. year built: {row['yr_built']}").add_to(
            marker_cluster)
    with c:
        folium_static(density_map)

    # Region Price Map
    geo_file.set_crs(epsg='4326', inplace=True)
    geo_file.to_crs('epsg:3857')

    c3.header('Price Density')
    data_2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    data_2.columns = ['ZIP', 'PRICE']
    geo_file = geo_file[geo_file['ZIP'].isin(data_2['ZIP'].to_list())]
    df1 = data_2.copy()

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom=15)

    region_price_map.choropleth(data=df1, geo_data=geo_file,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')
    with c3:
        folium_static(region_price_map)


def dashboard(data, built, dat):

    df_year = data.loc[df.yr_built > built][['price', 'yr_built']].groupby('yr_built').mean().reset_index()
    day = data.loc[data.date > dat][['price', 'date']].groupby('date').mean().reset_index()

    fig = plt.figure(figsize=(20, 12))

    specs = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.3)
    ax1 = fig.add_subplot(specs[0, :])
    ax2 = fig.add_subplot(specs[1, :])

    ax1.plot(day['date'], day['price'])
    ax1.set_title('Average Price by Day', fontdict={'fontsize': 21, 'family': 'Source Sans Pro', 'weight': 'bold'})
    ax1.set_xlabel('Day', fontdict={'fontsize': 15, 'family': 'Source Sans Pro'})
    ax1.set_ylabel('Average Price', fontdict={'fontsize': 15, 'family': 'Source Sans Pro'})
    ax1.ticklabel_format(axis='y', style='plain')

    ax2.plot(df_year['yr_built'], df_year['price'])
    ax2.set_title('Average Price by Year', fontdict={'fontsize': 21, 'family': 'Source Sans Pro', 'weight': 'bold'})
    ax2.set_xlabel('Year', fontdict={'fontsize': 15, 'family': 'Source Sans Pro'})
    ax2.set_ylabel('Average Price', fontdict={'fontsize': 15, 'family': 'Source Sans Pro'})
    plt.xticks(rotation=60)

    return fig


def commercial(data):
    # ===========
    # Dashboard
    # ============

    st.title('Commercial Attributes')

    # Filters
    st.sidebar.title('Commercial Options')
    min_yr_built = data['yr_built'].min()
    max_yr_built = data['yr_built'].max()
    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_value=min_yr_built, max_value=max_yr_built, value=min_yr_built)

    min_date = dt.datetime(year=2014, month=5, day=2)
    max_date = dt.datetime(year=2015, month=5, day=27)
    st.sidebar.subheader('Select Max Date')
    f_date = st.sidebar.slider('Date', min_value=min_date, max_value=max_date, value=min_date)
    dash = dashboard(data, built=f_year_built, dat=f_date)
    st.write(dash)
    return None


def price_dist(data):
    ###### Price Distribution ############

    st.sidebar.subheader('Select Max Price')
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    f_price = st.sidebar.slider('Price Range', min_value=min_price, max_value=max_price, value=min_price)

    st.subheader('Price Distribution')
    price_d = px.histogram(data[data['price'] < f_price], x='price', nbins=50)
    st.plotly_chart(price_d, use_container_width=True)
    return None


def attributes_distributions(data):
    #### Attributtes Distributions #####

    st.sidebar.header('Attirbutes Options')
    f_bed = st.sidebar.selectbox("Select Max Bedrooms", options=data.bedrooms.unique())
    f_bath = st.sidebar.selectbox("Select Max Bathrooms", options=data.bathrooms.unique())

    c1, c2 = st.columns(2)

    # Houses per bedrooms
    bed_dist = px.histogram(data[data.bedrooms < f_bed], x='bedrooms', nbins=19)
    c1.subheader('Bedrooms Distribution')
    c1.plotly_chart(bed_dist, use_container_width=True)

    # Houses Bathrooms
    bath_dist = px.histogram(data[data.bathrooms < f_bath], x='bathrooms', nbins=20)
    c2.subheader('Bathrooms Distribution')
    c2.plotly_chart(bath_dist, use_container_width=True)

    c1, c2 = st.columns(2)
    f_floors = st.sidebar.selectbox("Select Max Floors", options=data.bedrooms.unique())

    # Houses per floors
    floor_dist = px.histogram(data[data.floors < f_floors], x='floors', nbins=25)
    c1.subheader('Floor Distribution')
    c1.plotly_chart(floor_dist, use_container_width=True)

    ## Houses pe Waterview
    f_waterfront = st.sidebar.checkbox('Only houses with waterfront')
    if f_waterfront:
        df_1 = data[data['is_waterfront'] == 'Yes']
    else:
        df_1 = data.copy()
    c2.subheader('Water view Distribution')
    w_dist = px.histogram(df_1, x='waterfront', nbins=10)
    c2.plotly_chart(w_dist, use_container_width=True)
    return None


if __name__ == '__main__':

    # Get Data
    path = 'kc_houses.csv'
    ur = 'Zip_Codes.geojson'

    df = get_data(path)
    geofile = get_geofile(ur)

    ##### Transformation #####
    # Price by m²
    df = set_feature(df)
    df = null_values(df)

    # Data Overview
    data_overview(df)
    # Map plot
    portfolio_density(df, geofile)
    # Commercial Attributes
    commercial(df)
    price_dist(df)
    attributes_distributions(df)
