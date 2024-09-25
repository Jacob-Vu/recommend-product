import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the data
# We assume that 'sample_user_product_data.csv' is a CSV file with two columns: 'user_id' and 'product_id'.
# Each row represents a product bought by a specific user.
df = pd.read_csv('sample_user_product_data.csv')

# Step 2: Create a "user-product matrix"
# This matrix will show whether a user has bought a product or not.
# Rows represent users, columns represent products. 
# The value is 1 if the user has bought the product, and 0 if they haven't.
# 'pivot_table' converts the raw data into this matrix form.
user_product_matrix = df.pivot_table(index='user_id', columns='product_id', aggfunc='size', fill_value=0)

# Step 3: Calculate user similarity
# We'll compare how similar users are by looking at the products they bought.
# The similarity is calculated using cosine similarity, which is a measure of how similar two users are based on the products they've purchased.
# Cosine similarity outputs values between 0 and 1, where 1 means users are identical (i.e., they bought the same products).
user_similarity = cosine_similarity(user_product_matrix)

# Step 4: Convert the similarity matrix into a DataFrame for easier use.
# Rows and columns are both users, and the value at a position (i, j) tells us how similar user i is to user j.
user_similarity_df = pd.DataFrame(user_similarity, index=user_product_matrix.index, columns=user_product_matrix.index)

# Step 5: Get top N similar users
# This function finds the users who are most similar to the given user.
def get_top_n_similar_users(user_id, n=5):
    # We sort users based on their similarity to the target user, in descending order (from most similar to least similar).
    # We exclude the target user (user_id) from the results.
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n+1]
    return similar_users

# Step 6: Recommend products
# This function recommends products to the target user based on what similar users have bought.
def recommend_products(user_id, user_product_matrix, user_similarity_df, n_recommendations=5):
    # Get the most similar users to the target user.
    similar_users = get_top_n_similar_users(user_id, n=len(user_product_matrix))

    # Find out which products the target user has already bought.
    user_bought = user_product_matrix.loc[user_id]
    user_bought_products = user_bought[user_bought > 0].index.tolist()

    # Initialize an empty dictionary to store scores for each product.
    # We'll use this dictionary to score products based on how many similar users bought them.
    product_scores = {}

    # Loop through each similar user and check which products they bought.
    for similar_user, similarity_score in similar_users.items():
        similar_user_bought = user_product_matrix.loc[similar_user]
        
        # Now, loop through all the products bought by this similar user.
        for product_id in similar_user_bought[similar_user_bought > 0].index:
            # Only recommend products that the target user hasn't already bought.
            if product_id not in user_bought_products:
                # If the product isn't already in the product_scores dictionary, add it.
                if product_id not in product_scores:
                    product_scores[product_id] = 0
                # Add the similarity score of the similar user to this product's score.
                product_scores[product_id] += similarity_score
    
    # Sort the products by their scores (higher score means it's more likely a good recommendation).
    recommended_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    # Return the product IDs of the top N recommendations.
    return [(product, score) for product, score in recommended_products]

# Step 7: Example usage
# Here we use an example user, 'user_1', to get top 5 product recommendations for them.
user_id = 'user_1'  # The ID of the user we are generating recommendations for.

# Get the top 5 recommended products for the user.
recommended_products = recommend_products(user_id, user_product_matrix, user_similarity_df, n_recommendations=5)

# Output the result
print(f"Top 5 recommended products for {user_id}: {recommended_products}")
