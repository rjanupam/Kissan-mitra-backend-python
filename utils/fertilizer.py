import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def fertilizer_preprocess(single_data, model):
    df = pd.DataFrame(
        [single_data],
        columns=[
            "Temperature",
            "Humidity",
            "Moisture",
            "SoilType",
            "CropType",
            "Nitrogen",
            "Potassium",
            "Phosphorous",
        ],
    )

    encode_soil = LabelEncoder()
    encode_crop = LabelEncoder()

    df["SoilType"] = encode_soil.fit_transform(df["SoilType"])
    df["CropType"] = encode_crop.fit_transform(df["CropType"])

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), ["SoilType", "CropType"])],
        remainder="passthrough",
    )
    X = ct.fit_transform(df)

    sc = StandardScaler()
    X = sc.fit_transform(X)

    return pd.DataFrame(X, columns=model.feature_names_in_)
