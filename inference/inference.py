import json
from catboost import CatBoostRegressor
import streamlit as st
import pandas as pd
import numpy as np


## -----
def dataPreproc(df):
    # text NaN filling
    cols = ["Бренд", "Модель", "Тип машины", "Локация", "Цвет", "Тип кузова"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str)

    df = pd.get_dummies(df, columns=["Привод"], drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=["Топливо"], drop_first=True, dtype=int)
    df = pd.get_dummies(df, columns=["Тип кузова"], drop_first=True, dtype=int)

    df['Пробег'] = pd.to_numeric(df['Пробег'], errors='coerce').fillna(0)
    df["log_пробег"] = np.log1p(df["Пробег"])
    df["Полное название"] = df["Полное название"].astype(str)

    # synthetic features
    df["Возраст"] = 2026-df["Год выпуска"]
    df["Средне_годовой_пробег"] = df["Пробег"]/(df["Возраст"]+1)
    df["Оценка/год"] = df["Оценка эксперта"]/(df["Возраст"]+1)

    with open("models/columns.json", "r") as f:
        model_columns = json.load(f)
    
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# -----
st.set_page_config(
    page_title="Car Price Estimator",
    layout="wide"
)
# -----

# ----- lists for selectors
with open("inference/data/brands.json", "r", encoding="utf-8") as f:
    brands = json.load(f)
with open("inference/data/privod.json", "r", encoding="utf-8") as f:
    privod = json.load(f)
with open("inference/data/fuel.json", "r", encoding="utf-8") as f:
    fuel = json.load(f)
with open("inference/data/cartype.json", "r", encoding="utf-8") as f:
    cartype = json.load(f)
# -----


# ----- models loading
models = []
for i in range(0,5):
    model = CatBoostRegressor()
    model.load_model(f"models/model{i}.cbm")
    models.append(model)
# -----


# ----- UI
st.title("Оценка рыночной стоимости автомобиля")
st.write("---")

st.sidebar.header('Параметры автомобиля')
with st.sidebar.form("car_params"):

    with st.sidebar.expander("Основная информация", expanded=True):
        auto_brand = st.selectbox("Бренд автомобиля", brands)
        auto_fullname = st.text_input("Полное название")
        auto_type = st.selectbox("Тип кузова", cartype)
        auto_prod_year = st.number_input("Год выпуска", step=1)
        auto_location = st.text_input("Локация продажи")

    with st.sidebar.expander("Технические характеристики", expanded=True):
        auto_km = st.number_input("Пробег", step=1)
        auto_engine = st.number_input("Объем двигателя", step = 0.1)
        auto_cyl = st.number_input("Кол-во цилиндров", step=1)
        auto_privod = st.selectbox("Привод", privod)
        auto_fuel = st.selectbox("Топливо", fuel)
        auto_doors = st.number_input("Кол-во дверей", step=1)

    auto_score = st.sidebar.number_input("Оценка эксперта")    

    one_car_button = st.sidebar.button('Рассчитать стоимость по параметрам', use_container_width=True)

    st.sidebar.write("---")

st.sidebar.header("Загрузить таблицу (.csv)")
table = st.sidebar.file_uploader("", type=["csv"])
with open("storage/sample.csv", "rb") as file:
    st.sidebar.download_button(
        label="Скачать пример таблицы",
        data=file,
        file_name="sample.csv",
        mime="text/csv"
    ) 
# -----

# ----- predicting
if table is not None:
    df = pd.read_csv(table)
    dfx = dataPreproc(df)
    results = []
    for i in range(0,5):
        dfy = np.round(np.expm1(models[i].predict(dfx)),4)
        results.append(dfy)

    mean_res = np.mean(results)
    y = pd.Series(dfy, name="Цена")
    final_result = df.join(y)

    st.subheader('Результат:')
    st.table(final_result)

    csv_res = final_result.to_csv(index=False)

    save_table_button = st.download_button(
        label="Сохранить таблицу",
        data=csv_res,
        file_name="table.csv",
        mime="text/csv"
        )
        

if one_car_button:
    data = {
        "Бренд": auto_brand,
        "Полное название": auto_fullname,
        "Год выпуска": auto_prod_year,
        "Оценка эксперта": auto_score,
        "Пробег": auto_km,
        "Объем двигателя": auto_engine,
        "Цилиндры": auto_cyl,
        "Двери": auto_doors,
        "Тип кузова": auto_type,
        "Привод": auto_privod,
        "Топливо": auto_fuel,
        "Локация": auto_location
    }
    df = pd.DataFrame([data])
    df = dataPreproc(df)

    prices = []
    for i in range(0,5):
        pr = np.expm1(models[i].predict(df))
        prices.append(pr)

    price = np.mean(prices)

    st.subheader('Результат:')
    st.write(f'Предполагаемая цена: **{price:.2f}** у.е.')
# -----   
