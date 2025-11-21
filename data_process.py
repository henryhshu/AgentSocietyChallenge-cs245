# import json
import orjson
import logging
import uuid
import pandas as pd
from tqdm import tqdm
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_FILES_YELP = [
    'yelp_academic_dataset_business.json', 
    'yelp_academic_dataset_user.json', 
    'yelp_academic_dataset_review.json'
]

REQUIRED_FILES_AMAZON = [
    'Industrial_and_Scientific.csv', 
    'Musical_Instruments.csv', 
    'Video_Games.csv',
    'Industrial_and_Scientific.jsonl', 
    'Musical_Instruments.jsonl', 
    'Video_Games.jsonl',
    'meta_Industrial_and_Scientific.jsonl', 
    'meta_Musical_Instruments.jsonl', 
    'meta_Video_Games.jsonl'
]

REQUIRED_FILES_GOODREADS = [
    'goodreads_books_children.json', 
    'goodreads_reviews_children.json', 
    'goodreads_books_comics_graphic.json', 
    'goodreads_reviews_comics_graphic.json', 
    'goodreads_books_poetry.json', 
    'goodreads_reviews_poetry.json'
]

def check_required_files(input_dir):
    """Check if all required files exist in the input directory."""
    all_required_files = REQUIRED_FILES_YELP + REQUIRED_FILES_AMAZON + REQUIRED_FILES_GOODREADS
    missing_files = []
    
    for file in all_required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"- {file}")
        return False
    return True

def process_yelp_data(input_dir, output_dir):
    logging.info("Loading and processing Yelp data...")
    # Input file paths
    business_file = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    user_file = os.path.join(input_dir, 'yelp_academic_dataset_user.json')
    review_file = os.path.join(input_dir, 'yelp_academic_dataset_review.json')
    # Output file paths
    merged_item_file = os.path.join(output_dir, 'item.json')
    merged_review_file = os.path.join(output_dir, 'review.json')
    merged_user_file = os.path.join(output_dir, 'user.json')

    # Find top cities and related data
    logging.info("Finding top cities by reviews for Yelp...")
    top_cities = ['Philadelphia', 'Tampa', 'Tucson']

    # Process businesses
    business_ids = set()
    with open(business_file, 'r') as f, open(merged_item_file, 'a') as items:
        for line in f:
            obj = orjson.loads(line)

            # Filter businesses in top cities
            if obj.get("city") in top_cities:
                business_ids.add(obj["business_id"])

                # Ensure proper keys for item.json
                obj["item_id"] = obj.pop("business_id")
                obj["source"] = "yelp"
                obj["type"] = "business"

                items.write(orjson.dumps(obj).decode() + "\n")
    
    # Process reviews
    user_ids = set()
    with open(review_file, 'r') as f, open(merged_review_file, 'a') as reviews:
        for line in f:
            obj = orjson.loads(line)

            # Filter reviews by known businesses
            if obj.get("business_id") in business_ids:
                user_ids.add(obj["user_id"])

                # Ensure proper keys for review.json
                obj["source"] = "yelp"

                reviews.write(orjson.dumps(obj).decode() + "\n")
    
    # Process users
    with open(user_file, 'r') as f, open(merged_user_file, 'a') as users:
        for line in f:
            obj = orjson.loads(line)

            # Filter users by known reviews
            if obj["user_id"] in user_ids:
                # Ensure proper keys for user.json
                obj["source"] = "yelp"

                users.write(orjson.dumps(obj).decode() + "\n")

    logging.info("... Done processing Yelp data")


def process_amazon_data(input_dir, output_dir):
    logging.info("Loading and processing Amazon data...")
    # Input file paths
    rating_only_files = ['Industrial_and_Scientific.csv', 'Musical_Instruments.csv', 'Video_Games.csv']
    review_files = ['Industrial_and_Scientific.jsonl', 'Musical_Instruments.jsonl', 'Video_Games.jsonl']
    meta_files = ['meta_Industrial_and_Scientific.jsonl', 'meta_Musical_Instruments.jsonl', 'meta_Video_Games.jsonl']
    # Output file paths
    merged_item_file = os.path.join(output_dir, 'item.json')
    merged_review_file = os.path.join(output_dir, 'review.json')
    merged_user_file = os.path.join(output_dir, 'user.json')

    # Process rating only data
    user_ids = set()
    item_ids = set()
    for rating_only_file in rating_only_files:
        rating_only_file_path = os.path.join(input_dir, rating_only_file)

        for chunk in pd.read_csv(rating_only_file_path, chunksize=10000):
            user_ids.update(chunk["user_id"])
            item_ids.update(chunk["parent_asin"])
    
    # Process reviews to add to merged users and reviews
    written_users = set()
    for review_file in review_files:
        review_file_path = os.path.join(input_dir, review_file)

        with open(review_file_path, 'r') as f, open(merged_user_file, 'a') as users, open(merged_review_file, 'a') as reviews:
            for line in f:
                obj = orjson.loads(line)

                # Filter by known users and items
                if obj["user_id"] in user_ids and obj["parent_asin"] in item_ids:
                    # Ensure proper keys for user.json
                    if obj["user_id"] not in written_users:
                        users.write(orjson.dumps({
                            "user_id": obj["user_id"],
                            "source": "amazon"
                        }).decode() + "\n")
                        written_users.add(obj["user_id"]) # ensure user is only added once
                    
                    # Ensure proper keys for review.json
                    obj["sub_item_id"] = obj.pop("asin")
                    obj["item_id"] = obj.pop("parent_asin")
                    obj["stars"] = obj.pop("rating")
                    obj["review_id"] = str(uuid.uuid4())
                    obj["source"] = "amazon"
                    obj["type"] = "product"

                    reviews.write(orjson.dumps(obj).decode() + "\n")

    # Process meta data to add to merged items
    for meta_file in meta_files:
        meta_file_path = os.path.join(input_dir, meta_file)

        with open(meta_file_path, 'r') as f, open(merged_item_file, 'a') as items:
            for line in f:
                obj = orjson.loads(line)

                # Filter by known items
                if obj["parent_asin"] in item_ids:
                    # Ensure proper keys for item.json
                    obj["item_id"] = obj.pop("parent_asin")
                    obj["source"] = "amazon"
                    obj["type"] = "product"

                    items.write(orjson.dumps(obj).decode() + "\n")
    
    logging.info("... Done processing Amazon data")

def process_goodreads_data(input_dir, output_dir):
    logging.info("Loading and processing Goodreads data...")
    # Input file paths
    book_files = ['goodreads_books_children.json', 'goodreads_books_comics_graphic.json', 'goodreads_books_poetry.json']
    review_files = ['goodreads_reviews_children.json', 'goodreads_reviews_comics_graphic.json', 'goodreads_reviews_poetry.json']
    # Output file paths
    merged_item_file = os.path.join(output_dir, 'item.json')
    merged_review_file = os.path.join(output_dir, 'review.json')
    merged_user_file = os.path.join(output_dir, 'user.json')

    # Process books to add to items
    for book_file in book_files:
        book_file_path = os.path.join(input_dir, book_file)

        with open(book_file_path, 'r') as f, open(merged_item_file, 'a') as items:
            for line in f:
                obj = orjson.loads(line)

                # Ensure proper keys for item.json
                obj["item_id"] = obj.pop("book_id")
                obj["source"] = "goodreads"
                obj["type"] = "book"

                items.write(orjson.dumps(obj).decode() + "\n")
    
    # Process reviews to add to merged reviews and merged users
    user_ids = set()
    for review_file in review_files:
        review_file_path = os.path.join(input_dir, review_file)

        with open(review_file_path, 'r') as f, open(merged_review_file, 'a') as reviews, open(merged_user_file, 'a') as users:
            for line in f:
                obj = orjson.loads(line)
                
                if obj["user_id"] not in user_ids:
                    # Ensure proper keys for user.json
                    users.write(orjson.dumps({
                        "user_id": obj["user_id"],
                        "source": "goodreads"
                    }).decode() + "\n")
                    user_ids.add(obj["user_id"]) # Ensure user is only added once
                
                # Ensure proper keys for review.json
                obj["item_id"] = obj.pop("book_id")
                obj["stars"] = obj.pop("rating")
                obj["text"] = obj.pop("review_text")
                obj["source"] = "goodreads"
                obj["type"] = "book"

                reviews.write(orjson.dumps(obj).decode() + "\n")
        
    logging.info("... Done processing Goodreads data")

def main():
    """Main function with updated processing logic."""
    parser = argparse.ArgumentParser(description="Process multiple datasets for analysis.")
    parser.add_argument('--input_dir', required=True, help="Path to the input directory containing all dataset files.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory for saving processed data.")
    args = parser.parse_args()

    # Check required files
    if not check_required_files(args.input_dir):
        return

    process_yelp_data(args.input_dir, args.output_dir)
    process_amazon_data(args.input_dir, args.output_dir)
    process_goodreads_data(args.input_dir, args.output_dir)

    logging.info("Data processing completed successfully.")

if __name__ == '__main__':
    main()