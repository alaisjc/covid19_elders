import pandas as pd
import github
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
sns.set_style("white")

plt.style.use(['default'])
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2

def get_covid_19_data(date_param=None, *, min_cases=0, ref_country=None, add_map=None, round_precision=2):
    """ get_covid_19_data method

    Parameters
    ----------
    date_param : str
            The date presribed for the study. Time series retrieved if None
    min : int
            Minimum number of confirmed cases for a country for being considered
    ref_country : str
            The country which is assumed to display the best plausible death rate
    add_map : dict
            A country to corrected or new value mapping

    Returns
    -------
    DataFrame
        The covid 19 data and the covid 19 death rate data

    """

    for item in github.Github().get_repo("microsoft/Bing-COVID-19-Data").get_contents("data/"):
        if "csv" in item.name:  # Making the assumption (still ok up to now) that there is one csv only
            csv_name = item.name
    raw_data = pd.read_csv("https://raw.githubusercontent.com/microsoft/Bing-COVID-19-Data/master/data/"+csv_name)

    selection = ['Country_Region', 'Updated', 'Deaths', 'Recovered', 'Confirmed']

    # Selection and formating
    if date_param is not None:
        raw_data = raw_data[raw_data['Updated']==date_param].copy()
    
    data = raw_data[selection].copy()

    data.loc[:, 'Deaths'] = data['Deaths'].apply(lambda x: float(x) if x!=np.nan else 0)
    data.loc[:, 'Recovered'] = data['Recovered'].apply(lambda x: float(x) if x!=np.nan else 0)
    data.loc[:, 'Confirmed'] = data['Confirmed'].apply(lambda x: float(x) if x!=np.nan else 0)

    # Grouping by country (and date possibly)
    if date_param is None:
        data_gb = data.groupby(['Country_Region', 'Updated'])
    else:
        data_gb = data.groupby('Country_Region')
    
    data_gb_death = data_gb['Deaths'].sum().reset_index(name='deaths')
    data_gb_recovered = data_gb['Recovered'].sum().reset_index(name='recovered')
    data_gb_confirmed = data_gb['Confirmed'].sum().reset_index(name='confirmed')
    data_by_country_ = data_gb_death.merge(data_gb_recovered).merge(data_gb_confirmed)
    data_by_country_.set_index('Country_Region', drop=True, inplace=True)

    # Selecting the most impacted countries according to the confirmed cases numbers
    data_by_country_ = data_by_country_[data_by_country_['confirmed'] >= min_cases]

    # Computing the death rate in percentages as the deaths over the confirmed cases
    data_by_country_['death rate'] = (
        data_by_country_['deaths'] / data_by_country_['confirmed'] * 100.0
        )  # Division by 0 already handled
    data_by_country_.loc[:, 'death rate'] = data_by_country_['death rate'].apply(lambda x: round(x, round_precision))

    country_set = list(data_by_country_.index.get_level_values(0))

    if ref_country is not None and ref_country in country_set and date_param is not None:
        ref_death_rate = data_by_country_.loc[ref_country, 'death rate']
        data_by_country_ = data_by_country_[data_by_country_['death rate'] >= ref_death_rate]
    
    if add_map is not None and isinstance(add_map, dict) and date_param is not None:
        for country, death_rate in add_map.items():
            if country in data_by_country_.index:
                print("New death rate of " + str(death_rate) + "fulfilled for: " + country)
            data_by_country_.loc[country, 'death rate'] = death_rate

    return data_by_country_


def get_covid_19_data_france(min_cases=None):
    # Loading the opencovid19-fr dataset
    data_gouv_France = pd.read_csv("https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv")

    # Selecting the Ministère de la Santé dataset which includes the so-called EHPAD
    data_gouv_France_ephad = data_gouv_France[
                (data_gouv_France['granularite']=='pays') & 
                (data_gouv_France['source_type']=='ministere-sante')]

    # Selecting the Agences Régionales de Santé dataset
    data_gouv_France = data_gouv_France[
                (data_gouv_France['granularite']=='region') & 
                (data_gouv_France['source_type']=='agences-regionales-sante')]

    selection_set = ['date', 'maille_nom', 'deces', 'deces_ehpad', 'cas_confirmes', 'cas_confirmes_ehpad']

    # Ministère de la Santé dataset
    data_gouv_France_ephad = data_gouv_France_ephad[selection_set].copy()
    if min_cases is not None:
        data_gouv_France_ephad = data_gouv_France_ephad[data_gouv_France_ephad['cas_confirmes'] >= min_cases]
    
    # Getting things clearer ("deces" stands for "deces_hospital")
    data_gouv_France_ephad['deces_hospital'] = data_gouv_France_ephad['deces']
    data_gouv_France_ephad['deces'] = data_gouv_France_ephad['deces'] + data_gouv_France_ephad['deces_ehpad']

    # Computing the deaths over confirmed cases share
    data_gouv_France_ephad['death rate'] = (
        data_gouv_France_ephad['deces'] / data_gouv_France_ephad['cas_confirmes'] * 100.0
    )
    data_gouv_France_ephad['death rate (ehpad)'] = (
        data_gouv_France_ephad['deces_ehpad'] / data_gouv_France_ephad['cas_confirmes_ehpad'] * 100.0
    )
    data_gouv_France_ephad['death rate (hospital)'] = (
        data_gouv_France_ephad['deces_hospital'] / (
        data_gouv_France_ephad['cas_confirmes'] - data_gouv_France_ephad['cas_confirmes_ehpad']
        ) * 100.0
    )

    # Setting the date as the index and filling NA values according to the ffill method
    data_gouv_France_ephad.set_index('date', drop=True, inplace=True)
    data_gouv_France_ephad.index = pd.to_datetime(data_gouv_France_ephad.index)
    data_gouv_France_ephad.fillna(method='ffill', inplace=True)

    # Agences Régionales de Santé dataset
    data_gouv_France_ = data_gouv_France[selection_set].copy()
    data_gouv_France_gb = data_gouv_France_.groupby('date')
    data_gouv_France_deces = data_gouv_France_gb['deces'].sum().reset_index(name='deces')
    data_gouv_France_confirmes = data_gouv_France_gb['cas_confirmes'].sum().reset_index(name='cas_confirmes')
    data_gouv_France_all = data_gouv_France_deces.merge(data_gouv_France_confirmes)
    data_gouv_France_all = data_gouv_France_all[data_gouv_France_all['cas_confirmes'] >= min_cases]

    # Computing the deaths over confirmed cases share
    data_gouv_France_all['death rate'] = data_gouv_France_all['deces'] / data_gouv_France_all['cas_confirmes'] * 100.0

    # Setting the date as the index
    data_gouv_France_all.set_index('date', drop=True, inplace=True)
    data_gouv_France_all.index = pd.to_datetime(data_gouv_France_all.index)

    data_by_age = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/eceb9fb4-3ebc-4da3-828d-f5939712600a")

    return data_gouv_France_ephad, data_gouv_France_all, data_by_age


def get_elders_hosp_share(raw_data, *, age_set=None, rolling=1):
    if age_set is None:
        age_set = ['E']
    
    def get_data_by_age(_age_set=None):
        if _age_set is None:
            _age_set = ['0']
        data_by_age = raw_data[raw_data['sursaud_cl_age_corona'].isin(_age_set)]
        data_by_age.loc[:, 'date_de_passage'] = pd.to_datetime(data_by_age['date_de_passage'], format='%Y-%m-%d')
        data_by_age.set_index(['sursaud_cl_age_corona', 'dep', 'date_de_passage'], drop=True, inplace=True)
        data_by_age_ = data_by_age.groupby(level=[2]).sum()
        data_by_age_rolling = data_by_age_.rolling(rolling, min_periods=None).sum()
        
        return data_by_age_rolling
    
    data_ = get_data_by_age()
    data_elders = get_data_by_age(age_set)

    data_['elders corona hosp. share'] = data_['nbre_hospit_corona']
    data_['elders corona hosp. share'] = (
        data_elders['nbre_hospit_corona'] / data_['elders corona hosp. share'] * 100.0
    )

    return data_[['elders corona hosp. share']]


def configure_plotting(_ax, *, spines_set_exclusion=None, annotate=False, xaxis_visible=True, yaxis_visible=True):
    if annotate:
        for p in _ax.patches:
            _ax.annotate(str(p.get_height()), (p.get_x() * 1.0, p.get_height() + 1.0))

    if spines_set_exclusion is None:
        spines_set_exclusion = list()

    for p in spines_set_exclusion:
        _ax.axes.spines[p].set_visible(False)
    
    _ax.axes.xaxis.set_visible(xaxis_visible)
    _ax.axes.yaxis.set_visible(yaxis_visible)


def plot_death_rate(data, date, *, figsize=(23,3)):
    data.sort_values(by='death rate', ascending=False, inplace=True)

    ax = data['death rate'].plot.bar(
        title='deaths over confirmed cases (%) for countries with 10k confirmed cases or more (on ' + date + ', sources: MS Bing covid 19 dataset & Ministère de la Santé)', 
        figsize=figsize
    )

    configure_plotting(ax, spines_set_exclusion=['top', 'left', 'bottom', 'right'], annotate=True, yaxis_visible=False)

    plt.show()


def plot_covid_19_time_series(data, country_set, *, label='death rate', unit='', month_min=3, rolling_param=7, figsize=(23,7), plot_date_interval=1):
    plt.figure(figsize=figsize)

    for country in country_set:
        bing_data_country = data.loc[country].copy()
        bing_data_country = bing_data_country.set_index('Updated')
        bing_data_country.index = pd.to_datetime(bing_data_country.index)
        bing_data_country_ = bing_data_country[bing_data_country.index.month >= month_min]
        bing_data_country_ = bing_data_country_.rolling(rolling_param, min_periods=1).mean()

        # ax_country = plt.plot(bing_data_country_[label], label=country)[0]
        ax_country = sns.lineplot(data=bing_data_country_[label], label=country)

        configure_plotting(ax_country, spines_set_exclusion=['top', 'right'])
    
    if unit is not None:
        plt.title(label=label + ' (' + unit + ') over time (source: Microsoft Bing dataset)')
    else:
        plt.title(label=label + ' over time (source: Microsoft Bing dataset)')

    plt.xticks(rotation=90)
    plt.axes().xaxis.set_major_locator(mdates.DayLocator(interval=plot_date_interval))
    plt.legend(loc='upper left', frameon=False)
    plt.show()


def plot_hosp_share_France(data, dep_mapping, *, figsize = (15,7), month_min = 3, rolling_param=7, plot_date_interval=1):
    plt.figure(figsize=figsize)

    for key, dep_set in dep_mapping.items():
        data_by_age_ = data[data['dep'].isin(dep_set)].copy()
        data_elders_hosp_share = get_elders_hosp_share(data_by_age_, rolling=rolling_param)
        data_elders_hosp_share = data_elders_hosp_share[
            (data_elders_hosp_share.index.month>=month_min) & (data_elders_hosp_share.index.day>rolling_param)
        ]
        # plot_ = plt.plot(data_elders_hosp_share['elders corona hosp. share'], label=key)[0]
        plot_ = sns.lineplot(data=data_elders_hosp_share['elders corona hosp. share'], label=key)
        configure_plotting(plot_, spines_set_exclusion=['top', 'right'])

    plt.xticks(rotation=90)
    plt.axes().xaxis.set_major_locator(mdates.DayLocator(interval=plot_date_interval))
    plt.title(label='elders (>75 years old) corona hospitalization share (' + str(rolling_param) + ' days rolling sum)')
    # plt.legend(loc='bottom left', frameon=False)
    plt.show();


def plotting_figure_from_df(data, title='', *, figsize=None, legend=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    
    # ax = plt.plot(data, label=data.columns)[0]
    ax = sns.lineplot(data=data).set(title=title)[0]

    configure_plotting(ax, spines_set_exclusion=['top', 'right'])

    plt.xticks(rotation=90)
    plt.axes().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.title(label=title)
    if legend is None:
        plt.legend(frameon=False)
    else:
        plt.legend(loc=legend, frameon=False)
    
    plt.show()
