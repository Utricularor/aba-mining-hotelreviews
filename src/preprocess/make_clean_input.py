import pandas as pd

def make_clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["is phrase B a Contrary of phrase A? "])
    return df

if __name__ == "__main__":
    input_file_path = "data/input/Task 3 - Room (Silver) - Contrary(N)Body(P).csv"
    output_file_path = "data/input/Silver_Room_ContN_BodyP.csv"

    df = pd.read_csv(input_file_path)
    # df = make_clean_input(df)
    df.to_csv(output_file_path, index=False)