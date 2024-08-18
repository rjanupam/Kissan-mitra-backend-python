import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def yield_preprocess(data, encoder, scaler):
    df = pd.DataFrame(
        [data],
        columns=[
            "State_Name",
            "Season",
            "Crop",
            "N",
            "P",
            "K",
            "pH",
            "rainfall",
            "Area_in_hectares",
        ],
    )

    categorical_features = ["State_Name", "Season", "Crop"]
    X_categorical = one_hot_encoder.transform(df[categorical_features])
    X_categorical_df = pd.DataFrame(
        X_categorical,
        columns=one_hot_encoder.get_feature_names_out(categorical_features),
    )

    df = df.drop(columns=categorical_features)
    df = pd.concat([df, X_categorical_df], axis=1)

    numerical_features = [
        "N",
        "P",
        "K",
        "pH",
        "rainfall",
        "temperature",
        "Area_in_hectares",
    ]
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df
