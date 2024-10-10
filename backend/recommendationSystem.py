# import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt-tab')


def processData():
    # load the data
    data = pd.read_csv("../data/Hotel_Reviews.csv")
    data.head()

    # Replacing United Kingdom with UK
    data["Hotel_Address"] = data["Hotel_Address"].str.replace("United Kingdom", "UK")

    # extracting the country of the hotel from the hotel address
    data["country"] = data["Hotel_Address"].apply(lambda x: x.split(" ")[-1])

    # dropping unnecessary columns
    data.drop(
        [
            "Additional_Number_of_Scoring",
            "Review_Date",
            "Reviewer_Nationality",
            "Negative_Review",
            "Review_Total_Negative_Word_Counts",
            "Total_Number_of_Reviews",
            "Positive_Review",
            "Review_Total_Positive_Word_Counts",
            "Total_Number_of_Reviews_Reviewer_Has_Given",
            "Reviewer_Score",
            "days_since_review",
            "lat",
            "lng",
        ],
        axis=1,
        inplace=True,
    )

    # tags column contains strings of tags converting it to list and adding it to tag field
    def impute(column):
        column = column.iloc[0]
        if not isinstance(column, list):
            return "".join(literal_eval(column))
        else:
            return column

    # data.head()
    data["Tags"] = data[["Tags"]].apply(impute, axis=1)

    # converting the country and Tags column to lower case for simplicity
    data["country"] = data["country"].str.lower()
    data["Tags"] = data["Tags"].str.lower()

    return data


data = processData()


def getRecommendations(location, description):
    description = description.lower()
    description = word_tokenize(description)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    filtered = {word for word in description if word not in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemmatizer.lemmatize(fs))

    country = data[data["country"] == location.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    cos = []

    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i])
        temp_set = [word for word in temp_token if word not in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemmatizer.lemmatize(s))

        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))

    country["similarity"] = cos
    country = country.sort_values(by="similarity", ascending=False)
    country.drop_duplicates(subset="Hotel_Name", keep="first", inplace=True)
    country.sort_values("Average_Score", ascending=False, inplace=True)
    country.reset_index(inplace=True)

    return (
        country[["Hotel_Name", "Average_Score", "Hotel_Address"]]
        .head()
        .to_json(orient="records")
    )


if __name__ == "__main__":
    print("Recommendation System")
    print(getRecommendations("UK", "good service and food"))
