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

        self.bert_model = None

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
            max_reviews: int|None = 3,
            get_reviews_method: str = 'first',
            get_reviews_representative_bucket_size: int = 3
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

                    if max_reviews is not None and max_reviews > 0 and len(reviews) >= max_reviews:
                        break
            if len(reviews) > 0 and (item_id or user_id):
                if get_reviews_method == "representative_sample":
                    reviews = self._get_representative_item_review_sample(reviews, get_reviews_representative_bucket_size)
                elif get_reviews_method != 'first':
                    raise ValueError(f"Invalid get review method: {get_reviews_method}")
                    
        return reviews

    def _get_representative_item_review_sample(self, reviews: List[Dict], get_reviews_representative_bucket_size: int) -> List[Dict]:
        """
        Get a representative sample of reviews by partitioning into a grid of stars and length.
        """
        if not reviews:
            return []

        # Helper to get tertile thresholds
        def get_tertiles(values):
            if not values:
                return 0, 0
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            # Use 33rd and 66th percentiles
            p33 = sorted_vals[max(0, int(n * 0.333) - 1)] if n > 0 else 0
            p66 = sorted_vals[max(0, int(n * 0.666) - 1)] if n > 0 else 0
            # Adjust indices to be safer and arguably more standard for small N
            # For N=3: indices 0, 1, 2. 0.33*3=1. 0.66*3=2.
            # values[0] is low, values[1] is med, values[2] is high (roughly)
            p33 = sorted_vals[int(n * 0.33)]
            p66 = sorted_vals[int(n * 0.66)]
            return p33, p66

        # Extract values
        stars = [float(r.get('stars', 0)) for r in reviews]
        lengths = [len(r.get('text', '')) for r in reviews]

        s33, s66 = get_tertiles(stars)
        l33, l66 = get_tertiles(lengths)

        # Initialize 9 buckets
        # Keys: (star_category, length_category)
        # Categories: 0=Low/Short, 1=Medium, 2=High/Long
        buckets = {(s, l): [] for s in range(3) for l in range(3)}

        for r in reviews:
            # Determine star category
            s_val = float(r.get('stars', 0))
            if s_val <= s33: s_cat = 0 # Low
            elif s_val <= s66: s_cat = 1 # Medium
            else: s_cat = 2 # High

            # Determine length category
            l_val = len(r.get('text', ''))
            if l_val <= l33: l_cat = 0 # Short
            elif l_val <= l66: l_cat = 1 # Medium
            else: l_cat = 2 # Long

            buckets[(s_cat, l_cat)].append(r)

        # Collect results
        result = []
        for key in buckets:
            bucket = buckets[key]
            # Sort by 'useful' descending
            bucket.sort(key=lambda x: int(x.get('useful', 0) or 0), reverse=True)
            # Limit to sample size
            result.extend(bucket[:get_reviews_representative_bucket_size])

        return result

   
    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()
        self.index_env.close()