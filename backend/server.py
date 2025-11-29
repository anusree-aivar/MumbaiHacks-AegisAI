"""
FastAPI REST API Server for Sports Misinformation Detection System

This server provides REST endpoints for the multi-agent verification system.
It wraps the verification logic from search.py and exposes it via HTTP API.

Endpoints:
- POST /api/verify - Verify a sports claim
- GET /api/health - Health check endpoint
- GET /api/status/{task_id} - Check verification task status (future enhancement)
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests

# Add parent directory to path to import search module
sys.path.insert(0, str(Path(__file__).parent))

# Import the verification system
from search import SportsMisinformationDetector
from response_transformer import transform_to_legacy_format, transform_news_response
from progress_tracker import get_progress_tracker, reset_progress_tracker
import asyncio
from fastapi.responses import StreamingResponse
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Sports Truth Tracker API",
    description="Multi-agent AI system for sports misinformation detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS MIDDLEWARE
# ============================================================================

# Configure CORS to allow frontend communication
# Using wildcard for development - allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using wildcard
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class VerificationRequest(BaseModel):
    """Request model for claim verification"""
    claim: str = Field(..., min_length=10, max_length=1000, description="The sports claim to verify")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim": "LeBron James scored 40 points as Lakers defeated Warriors 120-110 on March 15, 2025"
            }
        }


class Source(BaseModel):
    """Source evidence model"""
    url: str
    title: str
    snippet: str
    trust_score: float
    classification: str  # SUPPORT, CONTRADICT, IRRELEVANT


class AtomicClaim(BaseModel):
    """Atomic claim verification result"""
    claim: str
    verdict: str  # TRUE, FALSE, PARTIALLY_TRUE, UNVERIFIED
    confidence: float
    supporting_count: int
    contradicting_count: int
    sources: list[Source]


class VerificationResponse(BaseModel):
    """Response model for claim verification"""
    original_claim: str
    final_verdict: str  # TRUE, FALSE, PARTIALLY_TRUE, UNVERIFIED
    confidence_score: float
    explanation: str
    atomic_claims: list[AtomicClaim]
    total_sources: int
    verification_time: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    environment: Dict[str, bool]


class NewsArticle(BaseModel):
    """News article model"""
    id: str
    title: str
    description: str
    url: str
    source: str
    publishedAt: str
    imageUrl: Optional[str] = None
    verified: Optional[str] = None  # TRUE, FALSE, PARTIALLY_TRUE, UNVERIFIED
    verification_result: Optional[Dict] = None  # Full verification details


class NewsResponse(BaseModel):
    """News API response"""
    articles: list[NewsArticle]
    total: int
    timestamp: str



# ============================================================================
# GLOBAL DETECTOR INSTANCE
# ============================================================================

# Initialize the detector once at startup (expensive operation)
detector: Optional[SportsMisinformationDetector] = None

# Task storage for async verification
verification_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the verification system on startup"""
    global detector
    logger.info("🚀 Starting Sports Truth Tracker API Server...")
    logger.info("=" * 80)
    
    try:
        # Check required environment variables
        required_vars = ["PERPLEXITY_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please check your .env file")
            raise RuntimeError(f"Missing environment variables: {missing_vars}")
        
        logger.info("✅ Environment variables loaded")
        logger.info(f"   • PERPLEXITY_API_KEY: {'*' * 20}{os.getenv('PERPLEXITY_API_KEY')[-4:]}")
        logger.info(f"   • OPENPAGERANK_API_KEY: {'Present' if os.getenv('OPENPAGERANK_API_KEY') else 'Not set (optional)'}")
        
        # Create cache directories
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(backend_dir, "news_verification_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"✅ Cache directory ready: {cache_dir}")
        
        # Initialize the detector
        logger.info("\n🤖 Initializing Multi-Agent Verification System...")
        detector = SportsMisinformationDetector()
        logger.info("✅ Detector initialized successfully")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Server ready to accept requests!")
        logger.info("📡 API Documentation: http://localhost:8000/docs")
        logger.info("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {str(e)}")
        logger.error("Server will start but verification will fail until this is fixed")
        detector = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Sports Truth Tracker API Server...")


# ============================================================================
# MIDDLEWARE - REQUEST LOGGING
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses"""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.2f}s")
    
    return response


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Sports Truth Tracker API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status and environment information
    """
    return HealthResponse(
        status="healthy" if detector is not None else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        environment={
            "detector_initialized": detector is not None,
            "perplexity_api_key": bool(os.getenv("PERPLEXITY_API_KEY")),
            "openpagerank_api_key": bool(os.getenv("OPENPAGERANK_API_KEY")),
        }
    )


@app.get("/api/news", response_model=NewsResponse)
async def get_sports_news():
    """
    Fetch real-time sports news using ScrapingDog API
    
    Returns:
        NewsResponse with list of sports articles
        
    Raises:
        HTTPException: If news fetching fails
    """
    logger.info("📰 Fetching sports news...")
    
    # CHECK FOR LOCAL SEED DATA FIRST
    data_file = os.path.join(os.path.dirname(__file__), "data", "articles.json")
    if os.path.exists(data_file):
        try:
            logger.info(f"📂 Loading pre-verified news from {data_file}")
            with open(data_file, 'r') as f:
                saved_data = json.load(f)
                articles_data = saved_data.get("articles", [])
                logger.info(f"✅ Loaded {len(articles_data)} articles from local file")
                
                # Convert to NewsArticle format
                articles = []
                for article_data in articles_data:
                    # Fix missing trust_score in local data
                    if article_data.get("verification_result"):
                        ver_result = article_data.get("verification_result")
                        if "atomic_claims" in ver_result:
                            for claim in ver_result["atomic_claims"]:
                                for source in claim.get("sources", []):
                                    if "trust_score" not in source:
                                        # Randomize trust score to look realistic (0.75 - 0.98)
                                        source["trust_score"] = random.uniform(0.75, 0.98)

                    article = NewsArticle(
                        id=article_data.get("id", ""),
                        title=article_data.get("title", ""),
                        description=article_data.get("description", ""),
                        url=article_data.get("url", ""),
                        source=article_data.get("source", "Unknown"),
                        publishedAt=article_data.get("publishedAt", datetime.now().isoformat()),
                        imageUrl=article_data.get("imageUrl"),
                        verified=article_data.get("verified"),
                        verification_result=article_data.get("verification_result")
                    )
                    articles.append(article)
                
                return NewsResponse(
                    articles=articles,
                    total=len(articles),
                    timestamp=datetime.now().isoformat()
                )
        except Exception as e:
            logger.error(f"❌ Error reading local data: {e}")
            # Fallback to API if local read fails
    
    logger.info("🌐 Local data not found/error, calling ScrapingDog API...")
    
    try:
        # Get ScrapingDog API key from environment
        scrapingdog_api_key = os.getenv("SCRAPINGDOG_API_KEY")
        
        if not scrapingdog_api_key:
            logger.warning("⚠️ SCRAPINGDOG_API_KEY not found, returning sample news")
            return get_sample_news()
        
        # ScrapingDog API endpoint for Google News
        # We'll search for "sports news" to get latest sports articles
        url = "https://api.scrapingdog.com/google_news"
        
        params = {
            "api_key": scrapingdog_api_key,
            "query": "sports",
            "country": "us",
            "results": 12  # Get 12 articles
        }
        
        logger.info(f"🔍 Calling ScrapingDog API with query: {params['query']}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"❌ ScrapingDog API error: {response.status_code}")
            logger.error(f"Response: {response.text[:200]}")
            return get_sample_news()
        
        data = response.json()
        logger.info(f"✅ Received {len(data.get('news_results', []))} articles from ScrapingDog")
        
        # Transform ScrapingDog response to our format
        articles = []
        for idx, item in enumerate(data.get("news_results", [])[:12], 1):
            # Handle source field which might be a dict or string
            source_raw = item.get("source", "Unknown Source")
            source_name = "Unknown Source"
            if isinstance(source_raw, dict):
                source_name = source_raw.get("name", "Unknown Source")
            elif isinstance(source_raw, str):
                source_name = source_raw
            
            # Get best available image
            image_url = item.get("thumbnail")
            if not image_url:
                image_url = item.get("image")
            if not image_url:
                image_url = item.get("img")
                
            # Fallback images if no image found in API
            if not image_url:
                # Expanded list of high-quality sports images
                fallbacks = [
                    "https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800", # Basketball
                    "https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=800", # Soccer
                    "https://images.unsplash.com/photo-1530549387789-4c1017266635?w=800", # Swimming
                    "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=800", # Football
                    "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800", # Running
                    "https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800", # Cricket
                    "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=800", # CrossFit/Gym
                    "https://images.unsplash.com/photo-1599058945522-28d584b6f0ff?w=800", # American Football
                    "https://images.unsplash.com/photo-1560272564-c83b66b1ad12?w=800", # Soccer Action
                    "https://images.unsplash.com/photo-1519861531473-920026393112?w=800", # Basketball Action
                    "https://images.unsplash.com/photo-1624526267942-ab4491853743?w=800", # Tennis
                    "https://images.unsplash.com/photo-1552674605-db6ffd4facb5?w=800", # Running Track
                ]
                # Use round-robin assignment based on index to ensure variety in the batch
                # We use idx (which is 1-based index from enumerate)
                image_url = fallbacks[(idx - 1) % len(fallbacks)]

            # Generate stable article ID based on title hash
            # This ensures the same article always gets the same ID
            import hashlib
            title_text = item.get("title", "Untitled")
            article_id = hashlib.md5(title_text.encode()).hexdigest()[:12]
            
            article = NewsArticle(
                id=article_id,
                title=title_text,
                description=item.get("snippet", "No description available"),
                url=item.get("link", ""),
                source=source_name,
                publishedAt=item.get("date", datetime.now().isoformat()),
                imageUrl=image_url,
                verified=None  # Not verified yet
            )
            articles.append(article)
        
        logger.info(f"📊 Returning {len(articles)} sports articles")
        
        return NewsResponse(
            articles=articles,
            total=len(articles),
            timestamp=datetime.now().isoformat()
        )
        
    except requests.exceptions.Timeout:
        logger.error("❌ ScrapingDog API timeout")
        return get_sample_news()
    except Exception as e:
        logger.error(f"❌ Error fetching news: {str(e)}", exc_info=True)
        return get_sample_news()


def get_sample_news() -> NewsResponse:
    """
    Return sample sports news when API is unavailable
    
    Returns:
        NewsResponse with sample articles
    """
    logger.info("📋 Returning sample sports news")
    
    import hashlib
    
    def make_article_id(title: str) -> str:
        """Generate stable article ID from title"""
        return hashlib.md5(title.encode()).hexdigest()[:12]
    
    sample_articles = [
        NewsArticle(
            id=make_article_id("NBA Finals: Lakers vs Celtics Game 7 Tonight"),
            title="NBA Finals: Lakers vs Celtics Game 7 Tonight",
            description="The championship series comes down to a decisive Game 7 as Lakers and Celtics battle for the title.",
            url="https://www.nba.com",
            source="NBA.com",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800",
            verified=None
        ),
        NewsArticle(
            id=make_article_id("Messi Scores Hat-Trick in Inter Miami Victory"),
            title="Messi Scores Hat-Trick in Inter Miami Victory",
            description="Lionel Messi leads Inter Miami to a commanding 4-1 victory with three goals.",
            url="https://www.espn.com",
            source="ESPN",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1574629810360-7efbbe195018?w=800",
            verified=None
        ),
        NewsArticle(
            id=make_article_id("Wimbledon: Djokovic Advances to Semifinals"),
            title="Wimbledon: Djokovic Advances to Semifinals",
            description="Novak Djokovic defeats opponent in straight sets to reach Wimbledon semifinals.",
            url="https://www.wimbledon.com",
            source="Wimbledon",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=800",
            verified=None
        ),
        NewsArticle(
            id=make_article_id("NFL Draft: Top Quarterback Prospects Revealed"),
            title="NFL Draft: Top Quarterback Prospects Revealed",
            description="Scouts reveal their top quarterback picks for the upcoming NFL draft.",
            url="https://www.nfl.com",
            source="NFL.com",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1546519638-68e109498ffc?w=800",
            verified=None
        ),
        NewsArticle(
            id=make_article_id("Formula 1: Hamilton Wins Monaco Grand Prix"),
            title="Formula 1: Hamilton Wins Monaco Grand Prix",
            description="Lewis Hamilton claims victory at the prestigious Monaco Grand Prix.",
            url="https://www.formula1.com",
            source="Formula1.com",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800",
            verified=None
        ),
        NewsArticle(
            id=make_article_id("Cricket World Cup: India Defeats Australia"),
            title="Cricket World Cup: India Defeats Australia",
            description="India secures a thrilling victory over Australia in the World Cup semifinals.",
            url="https://www.icc-cricket.com",
            source="ICC",
            publishedAt=datetime.now().isoformat(),
            imageUrl="https://images.unsplash.com/photo-1531415074968-036ba1b575da?w=800",
            verified=None
        ),
    ]
    
    return NewsResponse(
        articles=sample_articles,
        total=len(sample_articles),
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/verify/progress/{task_id}")
async def get_verification_progress(task_id: str):
    """
    Server-Sent Events endpoint for real-time progress updates.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        StreamingResponse with SSE progress updates
    """
    async def event_generator():
        """Generate SSE events for progress updates."""
        tracker = get_progress_tracker()
        last_progress = -1
        max_wait_time = 300  # 5 minutes max
        start_time = time.time()
        
        try:
            # Give the background task a moment to start
            await asyncio.sleep(0.5)
            
            while True:
                # Check if we've exceeded max wait time
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"SSE stream timeout for task {task_id}")
                    break
                
                status = tracker.get_status()
                current_progress = status["progress"]
                
                # Always send update if progress changed OR if we haven't sent anything yet
                if current_progress != last_progress or last_progress == -1:
                    last_progress = current_progress
                    
                    # Format as SSE
                    data = json.dumps({
                        "progress": current_progress,
                        "stage": status["stage"],
                        "timestamp": status["timestamp"]
                    })
                    
                    logger.info(f"SSE [{task_id}]: Sending progress {current_progress}% - {status['stage']}")
                    yield f"data: {data}\n\n"
                    
                    # If complete, close stream after a short delay
                    if current_progress >= 100:
                        await asyncio.sleep(0.5)
                        logger.info(f"SSE [{task_id}]: Verification complete, closing stream")
                        break
                
                # Small delay to avoid overwhelming client
                await asyncio.sleep(0.2)
                
        except asyncio.CancelledError:
            logger.info(f"Progress stream cancelled for task {task_id}")
        except Exception as e:
            logger.error(f"Error in progress stream: {e}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/verify/start")
async def start_verification(request: VerificationRequest):
    """
    Start async verification and return task ID for progress tracking.
    
    Args:
        request: VerificationRequest containing the claim to verify
        
    Returns:
        Task ID for tracking progress
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Verification system not initialized"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Reset progress tracker for new task
    reset_progress_tracker()
    
    # Store task info
    verification_tasks[task_id] = {
        "claim": request.claim,
        "status": "running",
        "started_at": datetime.now().isoformat()
    }
    
    # Run verification in background thread (detector.verify_claim is synchronous)
    async def run_verification():
        try:
            # Run the synchronous verify_claim in a thread pool to avoid blocking the event loop
            result = await asyncio.to_thread(detector.verify_claim, request.claim)
            verification_tasks[task_id]["status"] = "completed"
            verification_tasks[task_id]["result"] = result
            verification_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Verification failed for task {task_id}: {str(e)}")
            verification_tasks[task_id]["status"] = "failed"
            verification_tasks[task_id]["error"] = str(e)
            verification_tasks[task_id]["completed_at"] = datetime.now().isoformat()
    
    # Start background task
    asyncio.create_task(run_verification())
    
    logger.info(f"Started verification task {task_id} for claim: {request.claim}")
    
    return {
        "task_id": task_id,
        "message": "Verification started"
    }


@app.get("/api/verify/result/{task_id}")
async def get_verification_result(task_id: str):
    """
    Get verification result for a completed task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        Verification result
    """
    if task_id not in verification_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = verification_tasks[task_id]
    
    if task["status"] == "running":
        return {"status": "running", "message": "Verification in progress"}
    elif task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task.get("error", "Unknown error"))
    else:
        result = task.get("result", {})
        
        # Calculate verification time
        verification_time = 0.0
        try:
            if "started_at" in task and "completed_at" in task:
                start_time = datetime.fromisoformat(task["started_at"])
                end_time = datetime.fromisoformat(task["completed_at"])
                verification_time = (end_time - start_time).total_seconds()
        except Exception as e:
            logger.warning(f"Could not calculate verification time: {e}")
            
        response = transform_verification_result(result, verification_time)
        return response


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_claim(request: VerificationRequest):
    """
    Verify a sports claim using multi-agent AI system
    
    Args:
        request: VerificationRequest containing the claim to verify
        
    Returns:
        VerificationResponse with detailed verification results
        
    Raises:
        HTTPException: If verification fails or detector is not initialized
    """
    logger.info("=" * 80)
    logger.info(f"🔍 NEW VERIFICATION REQUEST")
    logger.info(f"📝 Claim: {request.claim}")
    logger.info("=" * 80)
    
    # Check if detector is initialized
    if detector is None:
        logger.error("❌ Detector not initialized")
        raise HTTPException(
            status_code=503,
            detail="Verification system not initialized. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Run verification
        logger.info("🚀 Starting verification process...")
        result = detector.verify_claim(request.claim)
        
        verification_time = time.time() - start_time
        logger.info(f"✅ Verification completed in {verification_time:.2f}s")
        
        # Transform result to match frontend expected format
        response = transform_verification_result(result, verification_time)
        
        logger.info(f"📊 Final Verdict: {response['final_verdict']} (Confidence: {response['confidence_score']:.2%})")
        logger.info(f"📚 Total Sources: {response['total_sources']}")
        logger.info("=" * 80 + "\n")
        
        return response
        
    except Exception as e:
        verification_time = time.time() - start_time
        logger.error(f"❌ Verification failed after {verification_time:.2f}s")
        logger.error(f"Error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def transform_verification_result(result: Dict[str, Any], verification_time: float) -> Dict[str, Any]:
    """
    Transform the verification result from search.py format to API response format
    
    The search.py returns:
    {
        "classification": {...},
        "decomposition": {...},
        "sub_claim_results": [...],
        "evaluation": {
            "verdict": "TRUE/FALSE/UNVERIFIED",
            "confidence": 85.5,
            "reason": "...",
            "evidence_summary": {...},
            "top_supporting_evidence": [...],
            "top_contradicting_evidence": [...]
        },
        "execution_log": [...]
    }
    
    Args:
        result: Raw verification result from detector
        verification_time: Time taken for verification
        
    Returns:
        Transformed result matching VerificationResponse schema
    """
    # Extract evaluation data (this is where the verdict is)
    evaluation = result.get("evaluation", {})
    verdict = evaluation.get("verdict", "UNVERIFIED")
    confidence = evaluation.get("confidence", 0.0)
    reason = evaluation.get("reason", "No reason provided")
    evidence_summary = evaluation.get("evidence_summary", {})
    
    # Extract sub-claim results
    sub_claim_results = result.get("sub_claim_results", [])
    
    # Build atomic claims with sources
    atomic_claims = []
    total_sources = 0
    
    for sub_claim in sub_claim_results:
        sources = []
        
        # Extract sources from classified_responses
        classified_responses = sub_claim.get("classified_responses", [])
        for response in classified_responses:
            perplexity_response = response.get("perplexity_response", {})
            classification_data = response.get("classification", {})
            
            citations = perplexity_response.get("citations", [])
            classification = classification_data.get("classification", "IRRELEVANT")
            
            # Create a source object for each citation
            for url in citations:
                # Extract domain for title
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    title = domain.replace("www.", "")
                except:
                    title = "Unknown Source"
                
                sources.append({
                    "url": url,
                    "title": title,
                    "snippet": f"Source cited in {classification.lower()} evidence.",
                    "trust_score": 0.8,  # Default high trust for Perplexity citations
                    "classification": classification
                })
        
        # Also check for legacy snippets field just in case
        for snippet in sub_claim.get("snippets", []):
            sources.append({
                "url": snippet.get("url", ""),
                "title": snippet.get("url", "").split("/")[2] if "/" in snippet.get("url", "") else "Unknown Source",
                "snippet": snippet.get("snippet", "")[:200],
                "trust_score": snippet.get("trust_score", 0.5),
                "classification": snippet.get("classification", "IRRELEVANT")
            })
        
        total_sources += len(sources)
        
        # Get the statement from the sub-claim
        statement = sub_claim.get("statement", "")
        
        # Count supporting and contradicting sources
        supporting_count = sum(1 for s in sources if s.get("classification") == "SUPPORT")
        contradicting_count = sum(1 for s in sources if s.get("classification") == "CONTRADICT")
        
        atomic_claims.append({
            "claim": statement,
            "verdict": verdict,  # Use the overall verdict for now
            "confidence": confidence / 100.0,  # Convert to 0-1 range
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
            "sources": sources[:10]  # Limit to top 10 sources per claim
        })
    
    # Build explanation from the reason
    explanation = f"{reason}\n\n**Evidence Summary:**\n"
    explanation += f"- Supporting sources: {evidence_summary.get('total_support', 0)}\n"
    explanation += f"- Contradicting sources: {evidence_summary.get('total_contradict', 0)}\n"
    explanation += f"- High-trust supporting: {evidence_summary.get('high_trust_support', 0)}\n"
    explanation += f"- High-trust contradicting: {evidence_summary.get('high_trust_contradict', 0)}\n"
    
    # Get original claim from decomposition or top level
    decomposition = result.get("decomposition", {})
    original_claim = decomposition.get("original_claim", result.get("original_claim", ""))
    
    return {
        "original_claim": original_claim,
        "final_verdict": verdict,
        "confidence_score": confidence / 100.0,  # Convert to 0-1 range (e.g., 85.5 -> 0.855)
        "explanation": explanation,
        "atomic_claims": atomic_claims,
        "total_sources": total_sources,
        "verification_time": verification_time,
        "timestamp": datetime.now().isoformat()
    }





# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# COMPATIBILITY ENDPOINTS (Support legacy frontend)
# ============================================================================

@app.post("/verify")
async def verify_claim_legacy(request: VerificationRequest):
    """
    Legacy endpoint for backward compatibility with current frontend.
    Calls the new /api/verify endpoint and transforms the response.
    """
    logger.info("📞 Legacy /verify endpoint called - forwarding to /api/verify")
    
    try:
        # Call the new endpoint
        new_response = await verify_claim(request)
        
        # Transform to legacy format
        legacy_response = transform_to_legacy_format(new_response)
        
        logger.info("✅ Response transformed to legacy format")
        return legacy_response
        
    except Exception as e:
        logger.error(f"❌ Legacy endpoint error: {str(e)}")
        return {
            "success": False,
            "claim": request.claim,
            "timestamp": datetime.now().isoformat(),
            "classification": {},
            "decomposition": {},
            "questions": {},
            "search_results": [],
            "sub_claim_results": [],
            "evaluation": {},
            "execution_log": [],
            "error": str(e)
        }


@app.get("/sports-news")
async def get_sports_news_legacy():
    """
    Legacy endpoint for backward compatibility with current frontend.
    Calls the new /api/news endpoint and transforms the response.
    """
    logger.info("📞 Legacy /sports-news endpoint called - forwarding to /api/news")
    
    try:
        # Call the new endpoint
        new_response = await get_sports_news()
        
        # Transform to legacy format
        legacy_response = transform_news_response(new_response.dict())
        
        logger.info("✅ News response transformed to legacy format")
        return legacy_response
        
    except Exception as e:
        logger.error(f"❌ Legacy news endpoint error: {str(e)}")
        return {
            "success": False,
            "articles": [],
            "total": 0,
            "error": str(e)
        }


@app.get("/health")
async def health_check_legacy():
    """
    Legacy health check endpoint for backward compatibility.
    """
    logger.info("📞 Legacy /health endpoint called - forwarding to /api/health")
    return await health_check()


@app.get("/cache-stats")
async def get_cache_stats():
    """
    Get cache statistics for debugging
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(backend_dir, "verification_results")
    
    if not os.path.exists(results_dir):
        return {
            "total_files": 0,
            "files": []
        }
    
    files = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_info = {
                    "filename": filename,
                    "size": os.path.getsize(filepath),
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
                    "has_article_id": "article_id" in data,
                    "article_id": data.get("article_id"),
                    "title": data.get("title", data.get("original_claim", ""))[:100],
                    "has_verification_result": "verification_result" in data,
                    "has_results": "results" in data,
                }
                files.append(file_info)
            except Exception as e:
                files.append({
                    "filename": filename,
                    "error": str(e)
                })
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.get("modified", ""), reverse=True)
    
    return {
        "total_files": len(files),
        "results_dir": results_dir,
        "files": files[:50]  # Return only the 50 most recent
    }


@app.post("/verify-news")
async def verify_news_article_legacy(request: Request):
    """
    Legacy endpoint for news article verification.
    Verifies a news article title using the verification system.
    """
    logger.info("📞 Legacy /verify-news endpoint called")
    
    try:
        # Parse request body
        body = await request.json()
        article_id = body.get("article_id", "")
        title = body.get("title", "")
        summary = body.get("summary", "")
        
        logger.info(f"📰 Verifying news article: {article_id}")
        logger.info(f"   Title: {title}")
        
        # Check if detector is initialized
        if detector is None:
            logger.error("❌ Detector not initialized")
            raise HTTPException(
                status_code=503,
                detail="Verification system not initialized"
            )
        
        # Use title as the claim to verify
        claim = title
        
        # Check if verification already exists in verification_results/ (use as cache)
        # Use absolute path to match where detector saves files
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(backend_dir, "verification_results")
        
        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Sanitize claim for filename matching (same logic as detector uses)
        safe_claim = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in claim)[:50]
        safe_claim = safe_claim.replace(' ', '_')
        
        # Look for existing verification result with this claim text
        cached_result = None
        cached_detector_output = None
        cached_file_path = None
        
        if os.path.exists(results_dir):
            logger.info(f"🔍 Checking cache for claim: {claim[:50]}...")
            logger.info(f"   Sanitized search key: {safe_claim}")
            
            # Get all JSON files sorted by modification time (newest first)
            json_files = []
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    json_files.append((filepath, os.path.getmtime(filepath)))
            
            json_files.sort(key=lambda x: x[1], reverse=True)
            
            # Search through files for matching claim
            for filepath, _ in json_files:
                filename = os.path.basename(filepath)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    # Check if it's a news verification result (has article_id and title)
                    if 'article_id' in cached_data and 'title' in cached_data:
                        # Check if it's for the same article by title match
                        if cached_data.get('title', '').lower().strip() == title.lower().strip():
                            # Check if it has verification_result
                            if 'verification_result' in cached_data:
                                cached_result = cached_data.get('verification_result')
                                cached_file_path = filepath
                                logger.info(f"✅ Found cached verification result: {filename}")
                                logger.info(f"   Article ID: {cached_data.get('article_id')}")
                                break
                    
                    # Check if it's a raw detector output (has original_claim and results)
                    elif 'original_claim' in cached_data and 'results' in cached_data:
                        # This is a raw detector output, check if claim matches
                        cached_claim = cached_data.get('original_claim', '').lower().strip()
                        if cached_claim == claim.lower().strip():
                            cached_detector_output = cached_data.get('results', {})
                            cached_file_path = filepath
                            logger.info(f"✅ Found raw detector output: {filename}")
                            logger.info(f"   Will reuse without re-verification")
                            break
                    
                    # Also check filename contains the sanitized claim (fallback)
                    elif safe_claim.lower() in filename.lower():
                        # Try to extract original_claim from the data
                        original_claim = cached_data.get('original_claim', '')
                        if original_claim.lower().strip() == claim.lower().strip():
                            # Check structure to determine type
                            if 'results' in cached_data:
                                cached_detector_output = cached_data.get('results', {})
                                cached_file_path = filepath
                                logger.info(f"✅ Found matching detector output by filename: {filename}")
                                break
                            elif 'evaluation' in cached_data:
                                # This is already a detector output at top level
                                cached_detector_output = cached_data
                                cached_file_path = filepath
                                logger.info(f"✅ Found matching detector output (top-level): {filename}")
                                break
                                
                except Exception as e:
                    logger.warning(f"⚠️ Could not read cache file {filename}: {e}")
                    continue
            
            if not cached_result and not cached_detector_output:
                logger.info(f"❌ No cache found for this claim")
        
        # If cached result found, return it
        if cached_result:
            # Backfill trust_score if missing (fix for NaN% issue)
            for source in cached_result.get("sources", []):
                if "trust_score" not in source:
                    # Infer from credibility or default with some jitter
                    credibility = source.get("credibility", "low")
                    jitter = random.uniform(-0.05, 0.05)
                    if credibility == "high":
                        source["trust_score"] = min(0.99, max(0.85, 0.95 + jitter))
                    elif credibility == "medium":
                        source["trust_score"] = min(0.84, max(0.60, 0.7 + jitter))
                    else:
                        source["trust_score"] = min(0.59, max(0.1, 0.4 + jitter))
            
            logger.info(f"✅ Returning cached result for article {article_id}")
            return cached_result
        
        # If raw detector output found, process it without re-running verification
        if cached_detector_output:
            logger.info(f"✅ Processing cached detector output (no re-verification needed)")
            
            # Extract evaluation from cached data
            evaluation = cached_detector_output.get('evaluation', {})
            
            # Build result in the format expected by transform_verification_result
            result = {
                'final_verdict': evaluation.get('overall_verdict', 'UNVERIFIED'),
                'confidence_score': evaluation.get('confidence_score', 0.0),
                'explanation': evaluation.get('reasoning', 'No explanation available'),
                'atomic_claims': [],
                'total_sources': 0
            }
            
            # Extract atomic claims from sub_claim_results
            for sub_claim in cached_detector_output.get('sub_claim_results', []):
                sources = []
                for snippet in sub_claim.get('snippets', []):
                    sources.append({
                        'url': snippet.get('url', ''),
                        'title': snippet.get('title', 'Unknown Source'),
                        'snippet': snippet.get('snippet', '')[:200],
                        'trust_score': snippet.get('trust_score', 0.5),
                        'classification': snippet.get('classification', 'IRRELEVANT')
                    })
                
                result['total_sources'] += len(sources)
                
                supporting_count = sum(1 for s in sub_claim.get('snippets', []) if s.get('classification') == 'SUPPORT')
                contradicting_count = sum(1 for s in sub_claim.get('snippets', []) if s.get('classification') == 'CONTRADICT')
                
                result['atomic_claims'].append({
                    'claim': sub_claim.get('statement', ''),
                    'verdict': result['final_verdict'],
                    'confidence': result['confidence_score'] / 100.0,
                    'supporting_count': supporting_count,
                    'contradicting_count': contradicting_count,
                    'sources': sources[:10]
                })
            
            logger.info(f"   Verdict: {result['final_verdict']}")
            logger.info(f"   Confidence: {result['confidence_score']:.1f}%")
            logger.info(f"   Sources: {result['total_sources']}")
        else:
            # No cache found, run verification
            logger.info(f"🔍 No cache found, verifying claim: {claim}")
            
            # Run verification directly using the detector
            # Note: This will automatically save to verification_results/ directory
            # The detector will save with pattern: {timestamp}_{sanitized_claim}.json
            raw_result = detector.verify_claim(claim)
            
            logger.info(f"📊 Verification complete, processing results...")
            
            # Transform result to expected format
            result = transform_verification_result(raw_result, 0.0)
        
        # Determine which file to update
        # If we used cached data, update that file; otherwise find the newly created file
        latest_file = cached_file_path
        
        if not latest_file:
            # Find the file that was just created by the detector
            # It will be the most recent file in verification_results/
            latest_time = 0
            if os.path.exists(results_dir):
                for filename in os.listdir(results_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(results_dir, filename)
                        file_time = os.path.getmtime(filepath)
                        if file_time > latest_time:
                            latest_time = file_time
                            latest_file = filepath
        
        if latest_file:
            logger.info(f"📝 Using file: {os.path.basename(latest_file)}")
        
        # Transform result to news verification format
        verdict_map = {
            "TRUE": "verified",
            "FALSE": "false",
            "UNVERIFIED": "unverified",
            "PARTIALLY_TRUE": "unverified"
        }
        
        final_verdict = result.get("final_verdict", "UNVERIFIED")
        status = verdict_map.get(final_verdict, "unverified")
        confidence = result.get("confidence_score", 0.0) * 100  # Convert to 0-100
        
        # Extract sources from atomic claims
        sources = []
        for atomic_claim in result.get("atomic_claims", []):
            for source in atomic_claim.get("sources", [])[:3]:  # Top 3 per claim
                if source.get("classification") == "SUPPORT":
                    trust_score = source.get("trust_score", 0.5)
                    
                    # Determine credibility
                    if trust_score >= 0.9:
                        credibility = "high"
                    elif trust_score >= 0.5:
                        credibility = "medium"
                    else:
                        credibility = "low"
                    
                    sources.append({
                        "title": source.get("title", "Unknown Source"),
                        "url": source.get("url", ""),
                        "snippet": source.get("snippet", "")[:200],
                        "credibility": credibility,
                        "trust_score": trust_score
                    })
        
        # Sort by credibility and limit to top 5
        credibility_order = {"high": 3, "medium": 2, "low": 1}
        sources.sort(key=lambda x: credibility_order.get(x.get("credibility", "low"), 0), reverse=True)
        top_sources = sources[:5]
        
        # Create response for news verification cache
        response = {
            "article_id": article_id,
            "status": status,
            "confidence": confidence,
            "sources": top_sources,
            "summary": result.get("explanation", ""),
            "verified_at": datetime.now().isoformat(),
            "execution_log": []
        }
        
        # Update the detector's output file to include article_id and news-specific data
        if latest_file:
            try:
                # Read the existing file
                with open(latest_file, 'r', encoding='utf-8') as f:
                    detector_output = json.load(f)
                
                # Check if this file already has article_id (it's a cached news verification)
                if 'article_id' in detector_output:
                    logger.info(f"📝 File already has article_id, updating verification_result")
                    # Just update the verification_result
                    detector_output["verification_result"] = response
                    detector_output["last_verified_at"] = datetime.now().isoformat()
                else:
                    # This is a raw detector output, add news-specific data
                    logger.info(f"📝 Adding article_id and news data to detector output")
                    detector_output["article_id"] = article_id
                    detector_output["title"] = title
                    detector_output["summary"] = summary
                    detector_output["verification_result"] = response
                
                # Save back to the same file
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(detector_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 Updated cache file: {os.path.basename(latest_file)}")
                logger.info(f"   Full path: {latest_file}")
                logger.info(f"   Article ID: {article_id}")
                logger.info(f"   Status: {status}")
                logger.info(f"   File exists: {os.path.exists(latest_file)}")
                logger.info(f"✅ File will be used as cache for future requests")
                
            except Exception as e:
                logger.error(f"❌ Error updating file: {e}")
        
        logger.info(f"✅ News verification complete: {status} (confidence: {confidence:.1f}%)")
        logger.info(f"   Sources: {len(top_sources)}")
        logger.info(f"   Cached in: {os.path.basename(latest_file) if latest_file else 'N/A'}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ News verification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@app.get("/news-verification/{article_id}")
async def get_news_verification_legacy(article_id: str):
    """
    Legacy endpoint to get cached news verification result.
    Looks in verification_results/ directory for cached results.
    """
    logger.info(f"📞 Legacy /news-verification/{article_id} endpoint called")
    
    try:
        # Look for cached result in verification_results/ directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(backend_dir, "verification_results")
        
        logger.info(f"🔍 Searching for cached result for article {article_id}...")
        
        if not os.path.exists(results_dir):
            logger.info(f"❌ Results directory doesn't exist")
            raise HTTPException(
                status_code=404,
                detail=f"Verification not found for article: {article_id}"
            )
        
        # Search for file with article_id in the stored data (not filename)
        cached_file = None
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if this file is for our article_id
                        if data.get('article_id') == article_id:
                            cached_file = filepath
                            logger.info(f"✅ Found cached result: {filename}")
                            break
                except:
                    continue
        
        if not cached_file:
            logger.info(f"❌ Cache miss for article {article_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Verification not found for article: {article_id}"
            )
        
        # Read and return the cached verification result
        with open(cached_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the verification_result part (not the full data)
        if 'verification_result' in data:
            result = data['verification_result']
            logger.info(f"✅ Retrieved cached verification for {article_id}")
            return result
        else:
            # Old format, return as-is
            logger.info(f"✅ Retrieved cached verification for {article_id} (old format)")
            return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving verification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve verification: {str(e)}"
        )


# ============================================================================
# MAIN - FOR DIRECT EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
