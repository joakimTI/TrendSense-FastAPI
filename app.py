from fastapi import FastAPI, HTTPException
import dill
from fastapi.middleware.cors import CORSMiddleware

# Assuming df is your DataFrame
def process_dataframe(df):
    # Group by category and sort by ratings to get top products in each category
    top_rated_by_category = (
        df.sort_values(by="ratings_count", ascending=False)
          .groupby("Category")
          .head(5)  # Adjust this number based on the number of top products you want per category
    )
    return top_rated_by_category.to_dict(orient="records")




# Load the similarity matrix from the downloaded file
with open("similarity_matrix.pkl", "rb") as f:
    similarity_data = dill.load(f)

# Initialize FastAPI
app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pickled function

cosine_sim = similarity_data['similarity_matrix']
product_ids = similarity_data['ids']


# Load your data
products = dill.load(open("products_list.pkl", "rb"))
sentiments = dill.load(open("sentiments.pkl", "rb"))
top_rated = process_dataframe(products)


# Define the root endpoint
@app.get("/")
async def read_root():
    top_rated_products = process_dataframe(products)
    return {"products": top_rated_products}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = products[products["id"] == product_id].to_dict(orient="records")
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product[0]

@app.get("/recommendations/{product_id}")
def get_recommendations(product_id: int, num_recommendations: int = 5):
    if product_id not in products['id'].values:
        raise HTTPException(status_code=404, detail="Product not found")
    
    recommendations = recommend_items(product_id, num_recommendations)
    return {"recommendations": recommendations}

@app.get("/reviews/{product_id}")
def get_reviews_by_sentiment(product_id: int):
    # Filter reviews for the specific product
    product_reviews = sentiments[sentiments["id"] == product_id][["review", "sentiment"]]
    
    if product_reviews.empty:
        raise HTTPException(status_code=404, detail="No reviews found for this product")
    
    # Group reviews by sentiment
    grouped_reviews = product_reviews.groupby("sentiment")["review"].apply(list).to_dict()
    
    return grouped_reviews
@app.get("/categories")
def get_categories():
    
    categories = products['Category'].unique().tolist()
    return {"categories": categories}

@app.get("/products_by_category/{category}")
def get_products_by_category(category: str):
    top_rated_by_category = (products.sort_values(by="ratings_count", ascending=False).groupby("Category").head(5))  # Adjust this number based on the number of top products you want per category)
    filtered_products = top_rated_by_category[top_rated_by_category['Category'] == category]
    return {"products": filtered_products.to_dict(orient="records")}

# Assuming df is your DataFrame
def process_dataframe(df):
    # Group by category and sort by ratings to get top products in each category
    top_rated_by_category = (
        df.sort_values(by="ratings_count", ascending=False)
          .groupby("Category")
          .head(20)  # Adjust this number based on the number of top products you want per category
    )
    return top_rated_by_category.to_dict(orient="records")


def recommend_items(product_id, num_recommendations=5):
    # Validate product_id
    if product_id not in product_ids:
        raise HTTPException(status_code=404, detail="Product not found")

    # Find index of the product in the precomputed matrix
    product_idx = list(product_ids).index(product_id)

    # Get similarity scores and sort
    sim_scores = cosine_sim[product_idx]
    top_indices = sim_scores.argsort()[::-1][1:num_recommendations + 1]  # Exclude self

    # Map indices back to product IDs
    recommended_ids = [product_ids[i] for i in top_indices]

    # Retrieve product details
    recommended_products = products[products['id'].isin(recommended_ids)].to_dict(orient='records')
    return recommended_products

