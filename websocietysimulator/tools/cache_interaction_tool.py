import logging
import os
import json
import lmdb
from typing import Optional, Dict, List, Iterator
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir

        # Create LMDB environments
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        self.user_env = lmdb.open(os.path.join(self.env_dir, "users"), map_size=4 * 1024 * 1024 * 1024)
        self.item_env = lmdb.open(os.path.join(self.env_dir, "items"), map_size=8 * 1024 * 1024 * 1024)
        self.review_env = lmdb.open(os.path.join(self.env_dir, "reviews"), map_size=32 * 1024 * 1024 * 1024)

        self.index_env = lmdb.open(os.path.join(self.env_dir, "review_index"), map_size=8 * 1024 * 1024 * 1024, max_dbs=3)
        self.user_index = self.index_env.open_db(b"user_index", dupsort=True)
        self.item_index = self.index_env.open_db(b"item_index", dupsort=True)

        # Initialize the database if empty
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the LMDB databases with data if they are empty."""
        # Initialize users
        with self.user_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for user in tqdm(self._iter_file('user.json')):
                        cursor.put(
                            user['user_id'].encode(),
                            json.dumps(user).encode()
                        )

        # Initialize items
        with self.item_env.begin(write=True) as txn:
            if not txn.stat()['entries']:
                with txn.cursor() as cursor:
                    for item in tqdm(self._iter_file('item.json')):
                        cursor.put(
                            item['item_id'].encode(),
                            json.dumps(item).encode()
                        )

        # Initialize reviews and their indices
        with self.review_env.begin(write=True) as review_txn, self.index_env.begin(write=True) as index_txn:
            if not review_txn.stat()['entries']:
                for review in tqdm(self._iter_file('review.json')):
                    # Store the review
                    review_txn.put(
                        review['review_id'].encode(),
                        json.dumps(review).encode()
                    )

                    # Update item reviews index (store only review_ids)
                    index_txn.put(
                        review["user_id"].encode(),
                        review["review_id"].encode(),
                        db = self.user_index,
                        dupdata=True
                    )

                    # Update user reviews index (store only review_ids)
                    index_txn.put(
                        review["item_id"].encode(),
                        review["review_id"].encode(),
                        db=self.item_index,
                        dupdata=True
                    )

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        with self.user_env.begin() as txn:
            user_data = txn.get(user_id.encode())
            if user_data:
                return json.loads(user_data)
        return None

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(
            self,
            item_id: Optional[str] = None,
            user_id: Optional[str] = None,
            review_id: Optional[str] = None,
            max_reviews: int = 3
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    return [json.loads(review_data)]
            return []

        reviews = []
        with self.review_env.begin() as review_txn, self.index_env.begin() as index_txn:
            if item_id:
                cursor = index_txn.cursor(db=self.item_index)
                key = item_id.encode()
            elif user_id:
                cursor = index_txn.cursor(db=self.user_index)
                key = user_id.encode()
            else:
                return []

            # Fetch complete review data for each review_id
            if cursor.set_key(key):
                for rid in cursor.iternext_dup():
                    rdata = review_txn.get(rid)
                    if rdata:
                        reviews.append(json.loads(rdata))

                    if len(reviews) >= max_reviews:
                        break

        return reviews

    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()
        self.index_env.close()