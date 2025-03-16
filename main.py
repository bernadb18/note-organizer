import pandas as pan
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

file = "zzoom.csv"

data = pan.read_csv(file)
#print(data.head())


notes = input("notes: ")
keypoint2 = input("keypoint")
keypoint1 = input("keypoint1: ")
topic = input("topic: ")


def correstnotes():
   
    data["text"] = data["Topic"] + " " + data["Key Points"] + " " + data["keypoints2"]

    
    LE = LabelEncoder()
    data["Subject"] = LE.fit_transform(data["Subject"])

    
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(data["text"])  
    y = data["Subject"]  

    
    model = LogisticRegression(max_iter=1000)
    
    
    cross_val_score(model, X, y, cv=5)  
    

    
    model.fit(X, y)

    
    new_data = [topic + " " + keypoint1 + " " + keypoint2]  
    new_data_tfidf = tfidf.transform(new_data)  
    
    
    prediction = model.predict(new_data_tfidf)
    predicted_subject = LE.inverse_transform(prediction)
    print("Predicted Subject: {predicted_subject[0]}")

    
    if predicted_subject[0] == "Physics":
        with open("Physics.txt", "a") as file:
            file.write(notes)
            file.write("\n")
            file.close()
    elif predicted_subject[0] == "Maths":
        with open("Maths.txt", "a") as file:  
            file.write("\n  notes: ")
            file.write(notes)
            
            file.close()
    else:
        with open("Other.txt", "a") as file:  
            file.write(notes)
            file.close()


correstnotes()
