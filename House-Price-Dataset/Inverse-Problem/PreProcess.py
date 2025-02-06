import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(file_path, target_column, features):

    df = pd.read_csv(file_path)
    X = df[features]
    y = df[target_column]

    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X)
    X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)

    def remove_outliers(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    columns_to_check = [col for col in X_standardized_df.columns]
    df_clean = remove_outliers(X_standardized_df, columns_to_check)
    df_clean[target_column] = y.loc[df_clean.index]
    return df_clean
