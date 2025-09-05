import numpy as np
import pandas as pd
import os


def transform_frame(df_to_transform: pd.DataFrame) -> pd.DataFrame:
    """Function for transforming initial dataframe by adding filling `Nan` values and making extra features"""
    newdf = df_to_transform.copy()
    newdf["is_description"] = newdf["description"].notna().astype(int)
    newdf["description"] = newdf["description"].fillna("")
    newdf["name_word_count"] = newdf["name_rus"].str.split().str.len()
    newdf["desc_word_count"] = newdf["description"].str.split().str.len()
    newdf["name_length"] = newdf["name_rus"].str.len()
    newdf["desc_length"] = newdf["description"].str.len()

    # Списки ключевых слов (их можно и нужно расширять)
    trigger_words_name = [
        "оригинал",
        "100% оригинал",
        "не китай",
        "гарантия",
        "скидка",
        "распродажа",
        "дешево",
        "акция",
    ]
    trigger_words_desc = [
        "высококачественный",
        "качественный аналог",
        "европейское качество",
        "проверено",
        "гарантия качества",
        "лучшая цена",
    ]

    # Функция для проверки вхождения любого слова из списка
    def contains_any(text, word_list):
        if pd.isna(text):
            return 0
        return int(any(word in text.lower() for word in word_list))

    newdf["name_has_any_trigger"] = newdf["name_rus"].apply(
        lambda x: contains_any(x, trigger_words_name)
    )
    newdf["desc_has_any_trigger"] = newdf["description"].apply(
        lambda x: contains_any(x, trigger_words_desc)
    )

    # Подсчет заглавных букв (кричащие названия)
    newdf["name_caps_count"] = newdf["name_rus"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    newdf["name_caps_ratio"] = newdf["name_caps_count"] / (newdf["name_length"] + 1)

    # Подсчет восклицательных и вопросительных знаков
    newdf["name_excl_quest_count"] = newdf["name_rus"].apply(
        lambda x: str(x).count("!") + str(x).count("?")
    )
    newdf["desc_excl_quest_count"] = newdf["description"].apply(
        lambda x: str(x).count("!") + str(x).count("?")
    )

    def unique_words_ratio(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)

    newdf["name_unique_ratio"] = newdf["name_rus"].apply(unique_words_ratio)
    newdf["desc_unique_ratio"] = newdf["description"].apply(unique_words_ratio)

    def is_brand_in_desc(row):
        brand = row["brand_name"]
        desc = row["description"]
        if pd.isna(brand) or pd.isna(desc):
            return 0

        return int(str(brand).lower() in str(desc).lower())

    newdf["is_brand_in_desc"] = newdf.apply(is_brand_in_desc, axis=1)

    columns_to_fill = [
        "rating_1_count",
        "rating_2_count",
        "rating_3_count",
        "rating_4_count",
        "rating_5_count",
        "comments_published_count",
        "photos_published_count",
        "videos_published_count",
        "GmvTotal7",
        "GmvTotal30",
        "GmvTotal90",
        "ExemplarAcceptedCountTotal7",
        "ExemplarAcceptedCountTotal30",
        "ExemplarAcceptedCountTotal90",
        "OrderAcceptedCountTotal7",
        "OrderAcceptedCountTotal30",
        "OrderAcceptedCountTotal90",
        "ExemplarReturnedCountTotal7",
        "ExemplarReturnedCountTotal30",
        "ExemplarReturnedCountTotal90",
        "ExemplarReturnedValueTotal7",
        "ExemplarReturnedValueTotal30",
        "ExemplarReturnedValueTotal90",
        "ItemAvailableCount",
    ]
    newdf[columns_to_fill] = newdf[columns_to_fill].fillna(0)
    columns_to_drop = [
        "GmvTotal7",
        "GmvTotal90",
        "ExemplarAcceptedCountTotal7",
        "ExemplarAcceptedCountTotal90",
        "OrderAcceptedCountTotal7",
        "OrderAcceptedCountTotal90",
        "ExemplarReturnedCountTotal7",
        "ExemplarReturnedCountTotal90",
        "ExemplarReturnedValueTotal7",
        "ExemplarReturnedValueTotal90",
        "ItemVarietyCount",
    ]
    newdf = newdf.drop(columns_to_drop, axis=1)
    return newdf


def feature_creator(df: pd.DataFrame) -> pd.DataFrame:
    """ "Function for modifying dataset by adding new features to it"""
    days = [30]
    for i in days:
        # Соотношение возвратов к продажам
        df[f"return_to_sales_ratio_{i}"] = np.where(
            df[f"item_count_sales{i}"] > 0,
            df[f"item_count_returns{i}"] / df[f"item_count_sales{i}"],
            0,
        )

        # Соотношение стоимости возвратов к GMV
        df[f"return_value_to_gmv_ratio_{i}"] = np.where(
            df[f"GmvTotal{i}"] > 0,
            df[f"ExemplarReturnedValueTotal{i}"] / df[f"GmvTotal{i}"],
            0,
        )

        # Доля подозрительных возвратов
        df[f"suspicious_return_ratio_{i}"] = np.where(
            df[f"item_count_returns{i}"] > 0,
            df[f"item_count_fake_returns{i}"] / df[f"item_count_returns{i}"],
            0,
        )

        # Конверсия принятых заказов в продажи (но тут фигня, что не для всех OrderAcceptedCountTotal > item_count_sales, но для большинства)
        df[f"order_to_sales_ratio_{i}"] = np.where(
            df[f"OrderAcceptedCountTotal{i}"] > 0,
            df[f"item_count_sales{i}"] / df[f"OrderAcceptedCountTotal{i}"],
            0,
        )

        # Средняя стоимость возврата
        df[f"avg_return_value_{i}"] = np.where(
            df[f"ExemplarReturnedCountTotal{i}"] > 0,
            df[f"ExemplarReturnedValueTotal{i}"] / df[f"ExemplarReturnedCountTotal{i}"],
            0,
        )

        # фичи по рейтингам
    df["rating_amount"] = df[
        [
            "rating_1_count",
            "rating_2_count",
            "rating_3_count",
            "rating_4_count",
            "rating_5_count",
        ]
    ].sum(axis=1)
    df["mean_rating"] = (
        df["rating_1_count"]
        + 2 * df["rating_2_count"]
        + 3 * df["rating_3_count"]
        + 4 * df["rating_4_count"]
        + 5 * df["rating_5_count"]
    ) / df["rating_amount"]

    df["rating_var"] = (
        df["rating_1_count"] * (1 - df["mean_rating"]) ** 2
        + df["rating_2_count"] * (2 - df["mean_rating"]) ** 2
        + df["rating_3_count"] * (3 - df["mean_rating"]) ** 2
        + df["rating_4_count"] * (4 - df["mean_rating"]) ** 2
        + df["rating_5_count"] * (5 - df["mean_rating"]) ** 2
    ) / df["rating_amount"]

    median_name_var = (
        df.groupby(by="CommercialTypeName4", as_index=False)
        .agg({"rating_var": "median"})
        .rename(columns={"rating_var": "median_name_var"})
    )
    df = df.merge(median_name_var, on="CommercialTypeName4", how="left")

    median_brand_var = (
        df.groupby(by="brand_name", as_index=False)
        .agg({"rating_var": "median"})
        .rename(columns={"rating_var": "median_brand_var"})
    )
    df = df.merge(median_brand_var, on="brand_name", how="left")

    df.loc[df["rating_var"].isna(), "rating_var"] = df.loc[
        df["rating_var"].isna(), "median_name_var"
    ]
    df.loc[df["rating_var"].isna(), "rating_var"] = df.loc[
        df["rating_var"].isna(), "median_brand_var"
    ]
    df["rating_var"].fillna(df["rating_var"].median(), inplace=True)

    df.drop(["median_name_var", "median_brand_var"], axis=1, inplace=True)

    data_gr = df.groupby(by=["CommercialTypeName4", "brand_name"], as_index=False).agg(
        {"PriceDiscounted": "median"}
    )
    data_gr.rename(columns={"PriceDiscounted": "median_price_discount"}, inplace=True)
    df = df.merge(data_gr, on=["CommercialTypeName4", "brand_name"], how="left")

    data_gr_name = df.groupby(by=["CommercialTypeName4"], as_index=False).agg(
        {"PriceDiscounted": "median"}
    )
    data_gr_name.rename(
        columns={"PriceDiscounted": "median_name_price_discount"}, inplace=True
    )
    df = df.merge(data_gr_name, on=["CommercialTypeName4"], how="left")

    df.loc[df["median_price_discount"].isna(), "median_price_discount"] = df.loc[
        df["median_price_discount"].isna(), "median_name_price_discount"
    ]
    df.drop(["median_name_price_discount"], axis=1, inplace=True)

    df["mean_price_30"] = df["GmvTotal30"] / (df["item_count_sales30"] + 1e6)

    ## оставляем только логарифм времени
    df["log_time_seller"] = np.log(df["seller_time_alive"] + 1)
    df["log_time_item"] = np.log(df["item_time_alive"] + 1)

    df.drop(["seller_time_alive", "item_time_alive"], axis=1, inplace=True)

    ## заполнение nan в столбце mean_rating
    median_name_rating = (
        df.groupby(by="CommercialTypeName4", as_index=False)
        .agg({"mean_rating": "median"})
        .rename(columns={"mean_rating": "median_name_rating"})
    )
    df = df.merge(median_name_rating, on="CommercialTypeName4", how="left")

    median_brand_rating = (
        df.groupby(by="brand_name", as_index=False)
        .agg({"mean_rating": "median"})
        .rename(columns={"mean_rating": "median_brand_rating"})
    )
    df = df.merge(median_brand_rating, on="brand_name", how="left")

    df.loc[df["mean_rating"].isna(), "mean_rating"] = df.loc[
        df["mean_rating"].isna(), "median_name_rating"
    ]
    df.loc[df["mean_rating"].isna(), "mean_rating"] = df.loc[
        df["mean_rating"].isna(), "median_brand_rating"
    ]
    df["mean_rating"].fillna(df["mean_rating"].mean(), inplace=True)

    df.drop(["median_name_rating", "median_brand_rating"], axis=1, inplace=True)

    df["brand_name"].fillna("unknown", inplace=True)

    df_new = df.drop(
        [
            "id",
            "rating_1_count",
            "rating_2_count",
            "rating_3_count",
            "rating_4_count",
            "rating_5_count",
            "ItemID",
        ],
        axis=1,
    )
    return df_new


def image_path(item_id: int, train_test: str = "train", path: str = None) -> str | None:
    """Function that creates path to the images of the product with id = `item_id` for train or test dataset"""
    if path is None:
        path = os.getcwd()
    final_path = os.path.join(
        path, f"ml_ozon_counterfeit_{train_test}_images", f"{item_id}.png"
    )
    if os.path.isfile(final_path):
        return final_path
    else:
        return ""
