import pandas as pd
import numpy as np

def get_covid_19_data(raw_data, date_param, *, min_cases=0, ref_country=None, add_map=None):
    """ get_covid_19_data method

    Parameters
    ----------
    raw_data : DataFrame
            Raw covid 19 data
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
        The death rate data

    """

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
    data_by_country_.loc[:, 'death rate'] = data_by_country_['death rate'].apply(lambda x: round(x, 1))

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


def plot_hosp_share_France(data, dep_mapping, *, figsize = (15,7), month_min = 3, rolling_param=7):
    plt.figure(figsize=figsize)

    for key, dep_set in dep_mapping.items():
        data_by_age_ = data_by_age[data_by_age['dep'].isin(dep_set)].copy()
        data_elders_hosp_share = get_elders_hosp_share(data_by_age_, rolling=rolling_param)
        data_elders_hosp_share = data_elders_hosp_share[
            (data_elders_hosp_share.index.month>=month_min) & (data_elders_hosp_share.index.day>rolling_param)
        ]
        plot_ = plt.plot(data_elders_hosp_share['elders corona hosp. share'], label=key)[0]
        for p in ['top', 'right']:
            plot_.axes.spines[p].set_visible(False)

    plt.xticks(rotation=90)
    plt.axes().xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.title(label='elders (>75 years old) corona hospitalization share (' + str(rolling_param) + ' days rolling sum)')
    plt.legend(loc='bottom left', frameon=False)
    plt.show();
