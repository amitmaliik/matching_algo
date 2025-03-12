# from pymongo import MongoClient
import numpy as np
from bson import ObjectId
import json
import logging
import pymongo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get database connection
def get_db_connection():
    try:
        # Replace with your actual connection string
        client = pymongo.MongoClient("")
        mydb = client[""]
        col = mydb[""]
        return col
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    # Ensure vectors are complete
    min_len = min(len(v1), len(v2))
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)

# Function to extract clean embeddings from MongoDB document
def extract_embeddings(doc):
    if not doc or 'embeddings' not in doc:
        return None
        
    user_embeddings = []
    
    for embed_item in doc['embeddings']:
        if 'embeddings' in embed_item and isinstance(embed_item['embeddings'], list):
            # Some embeddings are nested lists, some are direct
            embedding_vectors = embed_item['embeddings']
            if embedding_vectors and isinstance(embedding_vectors[0], list):
                user_embeddings.append({
                    'vector': embedding_vectors[0],  # Take the first vector
                    'title': embed_item.get('title', 'No Title'),
                    'url': embed_item.get('url', 'No URL'),
                    'category': embed_item.get('category', 'No Category')
                })
            else:
                user_embeddings.append({
                    'vector': embedding_vectors,
                    'title': embed_item.get('title', 'No Title'),
                    'url': embed_item.get('url', 'No URL'),
                    'category': embed_item.get('category', 'No Category')
                })
    
    return {
        'id': doc.get('_id', 'unknown'),
        'address': doc.get('address', 'unknown'),
        'embeddings': user_embeddings
    }

# Function to match a user with source embeddings
def match_user_with_source(user_data, source_data, top_n=5):
    if not user_data or not source_data:
        return []
        
    results = []
    
    for user_embed_item in user_data['embeddings']:
        user_vector = user_embed_item['vector']
        
        # Skip if vector is empty or None
        if not user_vector:
            continue
            
        similarities = []
        
        for source_embed_item in source_data['embeddings']:
            source_vector = source_embed_item['vector']
            
            # Skip if vector is empty or None
            if not source_vector:
                continue
                
            sim_score = cosine_similarity(user_vector, source_vector)
            
            similarities.append({
                'score': sim_score,
                'user_title': user_embed_item['title'],
                'user_url': user_embed_item['url'],
                'source_title': source_embed_item['title'],
                'source_url': source_embed_item['url'],
                'source_category': source_embed_item.get('category', 'No Category')
            })
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep only top N matches for this embedding
        top_matches = similarities[:top_n]
        results.extend(top_matches)
    
    # Sort all results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

# Function to calculate overall match score for a user
def calculate_match_score(matches, top_n=5):
    if not matches:
        return 0
        
    # Take top N matches
    top_matches = matches[:top_n]
    
    # Calculate weighted average (higher weight to better matches)
    weights = np.exp(-0.5 * np.arange(len(top_matches)))
    weights = weights / np.sum(weights)  # Normalize weights
    
    scores = [match['score'] for match in top_matches]
    weighted_score = np.sum(np.array(scores) * weights)
    
    return weighted_score

# Main function to match users from MongoDB
def match_users_from_mongodb(source_address="BRAVE", collection=None):
    # Connect to MongoDB using the provided connection function
    if collection is None:
        collection = get_db_connection()
    
    if collection is None:
        logger.error("Failed to connect to the database")
        return
    
    # Get the source document
    source_doc = collection.find_one({"address": source_address})
    
    if not source_doc:
        logger.error(f"Source with address {source_address} not found in database")
        return
        
    # Extract source embeddings
    # import ipdb; ipdb.set_trace()
    source_data = extract_embeddings(source_doc)
    
    if not source_data:
        logger.error(f"No valid embeddings found for source {source_address}")
        return
    
    # Get the target documents (4 users)
    # target_docs = list(collection.find({"address": {"$ne": source_address}}).limit(4))
    target_docs = list(collection.find({"address": {"$ne": source_address}}).limit(20))
    
    # Process each target user
    all_results = []
    
    for target_doc in target_docs:
        user_id = target_doc.get('address', str(target_doc.get('_id', 'unknown')))
        logger.info(f"Matching user {user_id} with source {source_address}...")
        
        # Extract user embeddings
        user_data = extract_embeddings(target_doc)
        
        if not user_data:
            logger.warning(f"No valid embeddings found for user {user_id}")
            continue
        
        # Match user with source
        matches = match_user_with_source(user_data, source_data)
        
        # Calculate overall match score
        match_score = calculate_match_score(matches)
        
        # Store results
        result = {
            'user_id': user_id,
            'match_score': match_score,
            'matches': matches[:10]  # Store only top 10 matches for brevity
        }
        all_results.append(result)
        
        # Log top 3 matches for this user
        logger.info(f"Overall match score: {match_score:.4f}")
        logger.info(f"Top 3 matches for user {user_id}:")
        for i, match in enumerate(matches[:3]):
            logger.info(f"{i+1}. Match score: {match['score']:.4f}")
            logger.info(f"   User content: {match['user_title']}")
            logger.info(f"   Source content: {match['source_title']} (Category: {match['source_category']})")
    
    # Sort users by match score
    all_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Log final rankings
    logger.info("\n--- Final User Rankings ---")
    for i, result in enumerate(all_results):
        logger.info(f"{i+1}. User {result['user_id']} - Score: {result['match_score']:.4f}")
    
    return all_results

# Extended function that can be called from a Flask app
def find_matches(user_id, limit=10, collection=None):
    """
    Find matches for a specific user from all other users in the database
    
    Args:
        user_id: The ID or address of the user to find matches for
        limit: Maximum number of matches to return
        collection: MongoDB collection (if None, will use get_db_connection())
        
    Returns:
        List of matches with scores
    """
    # Connect to MongoDB using the provided connection function
    if collection is None:
        collection = get_db_connection()
    
    if collection is None:
        logger.error("Failed to connect to the database")
        return {"error": "Database connection failed"}
    
    # Get the source user document
    user_doc = collection.find_one({"address": user_id})
    
    if not user_doc:
        # Try searching by ObjectId if address search fails
        try:
            user_doc = collection.find_one({"_id": ObjectId(user_id)})
        except:
            logger.error(f"User {user_id} not found in database")
            return {"error": f"User {user_id} not found in database"}
    
    if not user_doc:
        logger.error(f"User {user_id} not found in database")
        return {"error": f"User {user_id} not found in database"}
        
    # Extract user embeddings
    user_data = extract_embeddings(user_doc)
    
    if not user_data:
        logger.error(f"No valid embeddings found for user {user_id}")
        return {"error": f"No valid embeddings found for user {user_id}"}
    
    # Get all other users
    other_users = list(collection.find({
        "$or": [
            {"address": {"$ne": user_id}},
            {"_id": {"$ne": user_doc.get("_id")}}
        ]
    }))
    
    # Match with each other user
    match_results = []
    
    for other_user in other_users:
        other_id = other_user.get('address', str(other_user.get('_id', 'unknown')))
        
        # Extract other user embeddings
        other_data = extract_embeddings(other_user)
        
        if not other_data:
            continue
        
        # Match user with other user
        matches = match_user_with_source(user_data, other_data)
        
        # Calculate overall match score
        match_score = calculate_match_score(matches)
        
        match_results.append({
            'user_id': other_id,
            'match_score': match_score,
            'top_matches': matches[:3]  # Include top 3 specific content matches
        })
    
    # Sort by match score (descending)
    match_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Return top matches
    return match_results[:limit]

# Example of how to use this in a Flask app
"""
from flask import Flask, jsonify, request
import logging

app = Flask(__name__)

@app.route('/api/matches/<user_id>', methods=['GET'])
def get_matches(user_id):
    limit = int(request.args.get('limit', 10))
    matches = find_matches(user_id, limit=limit)
    return jsonify(matches)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
"""

if __name__ == "__main__":
    # Run the MongoDB matching function as a standalone script
    match_users_from_mongodb()
