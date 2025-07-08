import requests
from config import Config

class TMDBApi:
    def __init__(self):
        self.config = Config()
        self.base_url = self.config.TMDB_BASE_URL
        self.api_key = self.config.TMDB_API_KEY
        self.image_base_url = self.config.TMDB_IMAGE_BASE_URL
    
    def get_poster_by_imdb_id(self, imdb_id: str):
        try:
            if not imdb_id.startswith('tt'):
                imdb_id = f"tt{imdb_id}"
            
            endpoint = f"{self.base_url}/find/{imdb_id}"
            params = {
                "api_key": self.api_key,
                "external_source": "tconst"
            }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            poster_path = None
            if data.get("movie_results"):
                poster_path = data["movie_results"][0].get("poster_path")
            elif data.get("tv_results"):
                poster_path = data["tv_results"][0].get("poster_path")
            
            if poster_path:
                return f"{self.image_base_url}{poster_path}"
            return None
            
        except Exception as e:
            print(f"❌ TMDB API Error for IMDB ID {imdb_id}: {str(e)}")
            return None
    
    def get_multiple_posters_by_imdb(self, items: list):
        results = []
        for item in items:
            imdb_id = item.get('tconst')
            
            if imdb_id:
                poster_url = self.get_poster_by_imdb_id(imdb_id)
                item['poster_url'] = poster_url
            else:
                item['poster_url'] = None
                
            results.append(item)
            
        return results
