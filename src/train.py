# src/train.py

from data_prep import prepare_long_df
from model import train_logreg

from recommender import recommend, DEFAULT_PENALTY

def main():
    print("Loading and cleaning data...")
    long_df = prepare_long_df("data/raw/antibiotics.csv")

    print("Training model...")
    model, metrics = train_logreg(long_df)

    print("\nROC-AUC:", metrics["roc_auc"])
    print(metrics["report"])



    some_bacteria = long_df["bacteria"].dropna().iloc[0]
    print("TEST bacteria:", some_bacteria)

    rec = recommend(model, long_df, bacteria=some_bacteria, top_k=10, penalty_config=DEFAULT_PENALTY)
    print(rec)



if __name__ == "__main__":
    main()