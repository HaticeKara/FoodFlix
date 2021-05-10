from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def recotfidf(user_input,df):
    vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range=(1, 2), min_df = 0, stop_words = 'english')
    vectors_content = vectorizer.fit_transform(df['content'])
    vectors_product = vectorizer.transform([user_input])
    cosine_similarities = linear_kernel(vectors_product,vectors_content)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    similar_items = [(cosine_similarities[0][i], df['index'][i]) for i in similar_indices]
    return similar_indices

def recocount(user_input,df):
    cv=CountVectorizer()
    vectors_content=cv.fit_transform(df['content'])
    vectors_product=cv.transform([user_input])
    cosine_similarities=linear_kernel(vectors_product,vectors_content)
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    similar_items = [(cosine_similarities[0][i], df['index'][i]) for i in similar_indices]
    return similar_indices

def get_results(df, found):
    results = []

    for i in found:
        name = df.iloc[i].product_name
        brand = df.iloc[i].brands
        nutriscore = df.iloc[i].nutrition_grade_fr
#        allergens = df.iloc[i].allergens
        ingredients = df.iloc[i].ingredients_text
        vals = df.iloc[i][[
                "energy_100g", "fat_100g", "saturated-fat_100g",
                "carbohydrates_100g", "sugars_100g", "fiber_100g",
                "proteins_100g", "salt_100g"
            ]]
        rename = {
                "energy_100g": "énergie (en kj)",
                "fat_100g": "lipides",
                "saturated-fat_100g": "dont saturés",
                "carbohydrates_100g": "glucides",
                "sugars_100g": "dont sucres",
                "fiber_100g": "fibres",
                "proteins_100g": "protéines",
                "salt_100g": "sel",
            }
        vals = vals.rename(index=rename)
        results.append([name, brand, nutriscore, ingredients, vals])
    return results
