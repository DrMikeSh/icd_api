from flask import Flask, request, jsonify
import requests
import os
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from openai import OpenAI
from pinecone import Pinecone

app = Flask(__name__)

# Setup rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PICONE_API_KEY = os.getenv('PICONE_API_KEY')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')  

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_request(req):
    logger.info(f"Request Method: {req.method}")
    logger.info(f"Request Headers: {req.headers}")
    logger.info(f"Request Body: {req.data}")
    logger.info(f"Request Args: {req.args}")


# Simple token-based authentication
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        log_request(request)
        token = request.headers.get('Authorization')
        if token != ACCESS_TOKEN:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/get_results', methods=['POST'])
@require_auth
@limiter.limit("10 per minute")  # Rate limit for this endpoint
def get_results():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    
    # Input validation
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Text must be a non-empty string'}), 400

    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        client = OpenAI()
        # Call OpenAI API with timeout
        text = text.replace("\n", " ")
        openai_response = client.embeddings.create(input = [text], model="text-embedding-3-large", dimensions=1024).data[0].embedding
    except Exception as e:
        return jsonify({'error': f'Error calling OpenAI API: {str(e)}'}), 500
    
    try:
        # Call Picone API with timeout
        pc = Pinecone(api_key=PICONE_API_KEY)
        index = pc.Index("icd-index")
        picone_response= index.query(
        vector=openai_response,
        top_k=10)     

        # Check if the response is valid
        if not picone_response:
            raise ValueError("Empty response from Picone API")

    except Exception as e:
        logging.error(f"Picone API call failed: {e}")
        return jsonify({'error': 'Error calling Picone API'}), 500

    results = picone_response.matches
    ids = ''
    for result in results:
        ids += result.id + ', '
    ids  = ids[:-2]
    return jsonify({'ids': ids})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
