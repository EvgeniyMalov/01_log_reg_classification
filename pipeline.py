import dill
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression


import warnings
warnings.filterwarnings('ignore')


def transforms(sessions: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:

    """
    Обрабатываем исходные датасеты
    Группировка и агрегация исходных дф, слияние и разделение на признаки и целевую переменную
    Возвращаем матрицу "объект-признак" и вектор целевой переменной
    """

    import pandas as pd

    def df_merge(sessions: pd.DataFrame, hits_2: pd.DataFrame) -> pd.DataFrame:

        import pandas as pd

        df_merged = pd.merge(
            sessions, hits_2,
            left_on='session_id',
            right_on='session_id'
        )

        # Сделаем группировку по client_id и аггрегацию по остальным столбцам
        get_max = lambda x: pd.Series.mode(x)[0]
        df_merged_fin = df_merged.groupby('client_id').agg(entries=('session_id', 'nunique'),
                                                           utm_source=('utm_source', 'mean'),
                                                           utm_medium=('utm_medium', get_max),
                                                           device_category=('device_category', get_max),
                                                           device_brand=('device_brand', get_max),
                                                           device_screen_resolution=(
                                                           'device_screen_resolution', get_max),
                                                           device_browser=('device_browser', get_max),
                                                           geo_country=('geo_country', get_max),
                                                           geo_city=('geo_city', get_max), month=('month', 'mean'),
                                                           hit_count=('hit_count', 'mean'),
                                                           count_page=('count_page', 'mean'),
                                                           target=('target', 'sum')
                                                           )
        df_merged_fin.reset_index(inplace=True)

        # Уберем аномалии по следующему принципу:
        # entries - количество заходов до 50,
        # hit_count - количество действий - до 150
        # target - количество целевых действий - до 30
        # Уберем колонку client_id и target переведем в 0/1 (1 - 1 и больше целевых действий)
        df_final = df_merged_fin.drop(['client_id'], axis=1)
        df_final = df_final[(df_final['entries'] <= 50) & (df_final['hit_count'] <= 150) & (df_final['target'] <= 30)]
        df_final['target'] = df_final['target'].apply(lambda x: 0 if x < 1 else 1)

        #df_final.to_csv(r'data\pipe_df_final.csv', index=False)
        print('БД склеены')

        return df_final

    def divide(df_final: pd.DataFrame) -> pd.DataFrame:

        import pandas as pd

        X = df_final.drop(['target'], axis=1)
        y = df_final['target']

        #X.to_csv(r'data\pipe_X.csv', index=False)
        #y.to_csv(r'data\pipe_y.csv', index=False)
        print('Разбивка на Х у')
        return X, y

    sessions = sessions
    hits = hits

    sessions.drop_duplicates()
    # Удаляем колонки с пустыми и неинформативными значениями
    empty_cols = ['utm_keyword', 'device_os', 'device_model', 'utm_campaign', 'utm_adcontent', 'visit_time']
    sessions = sessions.drop(empty_cols, axis=1)

    # Делим трафик на органический и самые популярные неорганические каналы
    sessions['utm_medium'] = sessions['utm_medium'].apply(
        lambda x: 'organic' if x in ['organic', 'referral', '(none)'] else x)
    sessions['utm_medium'] = sessions['utm_medium'].apply(
        lambda x: x if x in ['organic', 'banner', 'cpc', 'cpm'] else 'other')

    # Рекламу в соц сетях разделим на да - 1, нет - 0
    commerce = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
                'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw',
                'gVRrcxiDQubJiljoTbGm']
    sessions['utm_source'] = sessions['utm_source'].apply(lambda x: 1 if x in commerce else 0)

    # Разделю на самые популярные и остальные
    sessions['device_brand'] = sessions['device_brand'].apply(
        lambda x: x if x in ['Apple', 'Samsung', 'Xiaomi', 'Huawei']
        else 'other')
    sessions['device_screen_resolution'] = sessions['device_screen_resolution'].apply(lambda x: x if x in
                                                                                                     ['414x896',
                                                                                                      '1920x1080',
                                                                                                      '375x812',
                                                                                                      '393x851'] else 'other')

    # Разделю на Хром, Сафари и остальные. Отдельно соберу Сафари в одну строку
    sessions['device_browser'] = sessions['device_browser'].apply(lambda x: 'Safari' if 'Safari' in
                                                                                        x else x)
    sessions['device_browser'] = sessions['device_browser'].apply(lambda x: x if x in ['Safari', 'Chrome'] else 'other')

    # Россия - 1, остальные - 0
    sessions['geo_country'] = sessions['geo_country'].apply(lambda x: x if x == 'Russia' else 'other')

    # Разделю на самые популярные и остальные
    cities = ['Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Krasnodar']
    sessions['geo_city'] = sessions['geo_city'].apply(lambda x: x if x in cities else 'other')

    # Дату визита разделю на месяц, так как все данные за один год.
    sessions['month'] = sessions['visit_date'].apply(lambda x: int(list(x.replace('-', ' ').split())[1]))
    sessions = sessions.drop(['visit_date'], axis=1)

    # drop_duplicates
    hits.drop_duplicates()

    # del empty cols
    empty_cols_2 = ['hit_time', 'hit_referer', 'event_label', 'event_value']
    hits = hits.drop(empty_cols_2, axis=1)

    # Переведу целевые действия в 0/1
    goal = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
            'sub_custom_question_submit_click',
            'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click']
    hits['event_action'] = hits['event_action'].apply(lambda x: 1 if x in goal else 0)

    # Проведу группировку по session_id и аггрегацию
    hits_2 = hits.groupby('session_id').agg(hit_count=('hit_type', 'count'),
                                            count_page=('hit_page_path', 'nunique'),
                                            target=('event_action', 'sum'))
    hits_2.reset_index(inplace=True)

    #sessions.to_csv(r'data\pipe_sessions.csv', index=False)
    #hits_2.to_csv(r'data\pipe_hits_2.csv', index=False)


    print('БД обработаны')

    df_final = df_merge(sessions, hits_2)

    return divide(df_final)


def x_transform(X: pd.DataFrame) -> pd.DataFrame:

    """
    Обработка столбцов матрицы X
    Числовые колонки - стандартизация
    Категориальные - ohe-кодирование
    """

    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import MinMaxScaler

    num_col = X.select_dtypes(include=['float64', 'int64']).columns.to_list()[1:]
    cat_col = X.select_dtypes(include=['object']).columns.to_list()

    ohe = OneHotEncoder(sparse=False)
    data = X[cat_col]
    ohe.fit(data)
    ohe_categorical = ohe.transform(data)
    X[ohe.get_feature_names_out()] = ohe_categorical
    X = X.drop(cat_col, axis=1)

    mm_scaler = MinMaxScaler()
    data = X[num_col]
    mm_scaler.fit(data)
    X[num_col] = pd.DataFrame(mm_scaler.transform(data), columns=num_col)

    X[['utm_source', 'month', 'hit_count', 'count_page']] = X[
        ['utm_source', 'month', 'hit_count', 'count_page']].fillna(
        X[['utm_source', 'month', 'hit_count', 'count_page']].mean())
    X = X.reset_index(drop=True)

    #X.to_csv(r'data\pipe_X_trans.csv', index=False)
    print('Х трансформирован')

    return X


def pipeline() -> None:
    # with open('data/ga_sessions.pkl', 'rb') as file:
    #     sessions = pickle.load(file)
    # with open('data/ga_hits-001.pkl', 'rb') as file:
    #     hits = pickle.load(file)
    sessions = pd.read_csv('data/ga_sessions.csv')
    hits = pd.read_csv('data/ga_hits-001.csv')

    print('Прошла загрузка БД')

    X, y = transforms(sessions, hits)

    model = LogisticRegression()
    pipe = Pipeline([
        ('x_transform', FunctionTransformer(x_transform)),
        ('model', model)
     ])

    result = pipe.fit(X, y)

    with open('model/fin_model.pkl', 'wb') as file:
        dill.dump(result, file)

    print('Завершение работы')


if __name__ == '__main__':
    pipeline()