import asyncio
import math
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from google import genai
from google.cloud import firestore
from pydantic import BaseModel

load_dotenv()

# Define the model once as a module-level constant MODEL
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

app = FastAPI(title="Goldmine", description="Goldmine Discovery and Intelligence Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vertex AI client helper
_genai_client: Optional[genai.Client] = None

def get_vertex_client() -> genai.Client:
    """Initialise with Vertex AI only. If initialisation fails, raise error."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI initialization.")
    
    _genai_client = genai.Client(
        vertexai=True,
        project=project,
        location="europe-west1",
    )
    return _genai_client


# Persistence: Local file-backed storage with optional Firestore integration
DATA_STORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_store.json")
_firestore_db: Optional[firestore.Client] = None
_firestore_available: bool = True
_in_memory_runs: Dict[str, Dict[str, Any]] = {}
_in_memory_businesses: Dict[str, Dict[str, Any]] = {}


def _load_local_store():
    global _in_memory_runs, _in_memory_businesses
    try:
        if os.path.exists(DATA_STORE_FILE):
            with open(DATA_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                _in_memory_runs = data.get("runs", {})
                _in_memory_businesses = data.get("businesses", {})
    except Exception:
        pass


def _save_local_store():
    try:
        with open(DATA_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "runs": _in_memory_runs,
                "businesses": _in_memory_businesses
            }, f)
    except Exception:
        pass


# Initialize local store on startup
_load_local_store()


def get_firestore() -> Optional[firestore.Client]:
    global _firestore_db, _firestore_available
    if not _firestore_available:
        return None
    if _firestore_db is not None:
        return _firestore_db
    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        _firestore_db = firestore.Client(project=project, database="(default)")
        return _firestore_db
    except Exception:
        _firestore_available = False
        return None


def db_save_run(run_id: str, data: Dict[str, Any]):
    global _firestore_available
    _in_memory_runs[run_id] = {**_in_memory_runs.get(run_id, {}), **data}
    _save_local_store()
    
    db = get_firestore()
    if db:
        try:
            db.collection("runs").document(run_id).set(data, merge=True)
        except Exception as e:
            # If permission denied or forbidden, disable firestore to prevent repeated errors
            if "403" in str(e) or "permission" in str(e).lower():
                _firestore_available = False


def db_get_run(run_id: str) -> Optional[Dict[str, Any]]:
    global _firestore_available
    db = get_firestore()
    if db:
        try:
            doc = db.collection("runs").document(run_id).get()
            if doc.exists:
                remote_data = doc.to_dict() or {}
                # merge with in-memory if available
                return {**_in_memory_runs.get(run_id, {}), **remote_data}
        except Exception as e:
            if "403" in str(e) or "permission" in str(e).lower():
                _firestore_available = False
    return _in_memory_runs.get(run_id)


def db_save_business(place_id: str, data: Dict[str, Any]):
    global _firestore_available
    _in_memory_businesses[place_id] = data
    _save_local_store()
    
    db = get_firestore()
    if db:
        try:
            db.collection("businesses").document(place_id).set(data)
        except Exception as e:
            if "403" in str(e) or "permission" in str(e).lower():
                _firestore_available = False


def db_get_index_stats() -> Dict[str, int]:
    global _firestore_available
    total_businesses = len(_in_memory_businesses)
    categories = set(b.get("category") for b in _in_memory_businesses.values() if b.get("category"))
    
    db = get_firestore()
    if db:
        try:
            docs = db.collection("businesses").stream()
            db_categories = set()
            count = 0
            for doc in docs:
                count += 1
                cat = doc.to_dict().get("category")
                if cat:
                    db_categories.add(cat)
            if count > total_businesses:
                total_businesses = count
                categories = categories.union(db_categories)
        except Exception as e:
            if "403" in str(e) or "permission" in str(e).lower():
                _firestore_available = False
            
    return {
        "total_businesses": total_businesses,
        "category_count": len(categories)
    }


# 15 Fixed Categories
ALL_CATEGORIES = [
    "Restaurants and cafés",
    "Bars and pubs",
    "Beauty and aesthetics",
    "Hair",
    "Fitness",
    "Dental",
    "Private healthcare",
    "Veterinary",
    "Legal",
    "Accountancy",
    "Estate agencies",
    "Retail",
    "Home services",
    "Automotive",
    "Hospitality"
]

LONDON_BOROUGHS_AND_DISTRICTS = [
    "City of London", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham",
    "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", "Merton",
    "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
    "Soho", "Covent Garden", "Mayfair", "Fitzrovia", "Marylebone", "Bloomsbury", "Holborn",
    "Spitalfields", "Shoreditch", "Hoxton", "Clerkenwell", "Farringdon", "Bermondsey",
    "Brixton", "Clapham", "Battersea", "Balham", "Tooting", "Wimbledon", "Putney",
    "Dulwich", "Peckham", "Camberwell", "Elephant and Castle", "Vauxhall", "Kennington",
    "Blackheath", "Deptford", "Canary Wharf", "Isle of Dogs", "Stratford", "Dalston",
    "Stoke Newington", "Angel", "Highbury", "Finsbury Park", "Camden Town", "Primrose Hill",
    "Belsize Park", "Hampstead", "Highgate", "Kentish Town", "King's Cross", "Kings Cross",
    "St Pancras", "Euston", "Paddington", "Notting Hill", "Bayswater", "Holland Park",
    "Shepherd's Bush", "Acton", "Chiswick", "Brentford", "Twickenham", "Wembley",
    "Kilburn", "Maida Vale", "St John's Wood", "Swiss Cottage", "Golders Green",
    "Finchley", "Muswell Hill", "Crouch End", "Wood Green", "Tottenham", "Walthamstow",
    "Leyton", "Leytonstone", "Wanstead", "Bow", "Mile End", "Whitechapel", "Stepney",
    "Bethnal Green", "Wapping", "Limehouse", "Poplar", "Docklands", "Woolwich", "Eltham",
    "Catford", "Brockley", "New Cross", "Forest Hill", "Sydenham", "Crystal Palace",
    "Norwood", "Streatham", "Morden", "Surbiton", "Teddington", "Barnes", "Mortlake",
    "Kew", "Bexleyheath", "Romford", "Ilford", "Hammersmith", "Fulham", "Kensington",
    "Chelsea", "Richmond", "Kingston", "Knightsbridge", "Belgravia", "Pimlico", "Victoria",
    "Piccadilly", "Chinatown", "Seven Dials", "Whitehall", "Barbican", "Monument",
    "Blackfriars", "South Bank", "Waterloo", "London Bridge", "Borough", "Rotherhithe",
    "Surrey Quays", "Canada Water", "North Greenwich", "Charlton", "Plumstead", "Abbey Wood",
    "Sidcup", "Chislehurst", "Orpington", "Beckenham", "Penge", "Herne Hill", "Stockwell",
    "Mitcham", "Colliers Wood", "Raynes Park", "New Malden", "Hanwell", "Southall",
    "Hayes", "Uxbridge", "Ruislip", "Northwood", "Pinner", "Eastcote", "Stanmore",
    "Edgware", "Mill Hill", "Colindale", "Hendon", "East Finchley", "North Finchley",
    "Cockfosters", "Southgate", "Palmers Green", "Winchmore Hill", "Edmonton",
    "Hackney Wick", "Homerton", "Haggerston"
]


def extract_locality(address: str, fallback_location: str = "London") -> str:
    """Derive locality from the address by taking the London district or borough component if present, otherwise London."""
    if not address or not isinstance(address, str):
        return "London"

    sorted_areas = sorted(LONDON_BOROUGHS_AND_DISTRICTS, key=len, reverse=True)
    for area in sorted_areas:
        pattern = r"\b" + re.escape(area) + r"\b"
        if re.search(pattern, address, re.IGNORECASE):
            return area

    return "London"


def compute_percentile_ranks(values: List[float]) -> List[float]:
    """Compute percentile rank for a list of values (0.0 to 100.0)."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [100.0]
    
    ranks: List[float] = []
    for v in values:
        count_less = sum(1 for x in values if x < v)
        count_equal = sum(1 for x in values if x == v)
        pct = ((count_less + 0.5 * count_equal) / n) * 100.0
        ranks.append(round(pct, 1))
    return ranks


async def fetch_places_paginated(category: str, location: str, max_pages: int = 3) -> List[Dict[str, Any]]:
    """
    Page through Places with pageToken twice to collect up to 60 businesses.
    Sleep 2 seconds between page calls.
    """
    load_dotenv()
    maps_api_key = os.environ.get("MAPS_API_KEY")
    if not maps_api_key:
        raise RuntimeError("MAPS_API_KEY environment variable is not configured.")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": maps_api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.rating,places.userRatingCount,places.websiteUri,places.formattedAddress,nextPageToken",
    }

    collected: List[Dict[str, Any]] = []
    seen_ids = set()
    page_token: Optional[str] = None
    query = f"{category} in {location}" if location else category

    for page_idx in range(max_pages):
        if page_idx > 0:
            await asyncio.sleep(2)

        payload: Dict[str, Any] = {"textQuery": query}
        if page_token:
            payload["pageToken"] = page_token

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"Places API returned {resp.status_code} on page {page_idx}: {resp.text}")
                break
            data = resp.json()

        raw_places = data.get("places", [])
        if not isinstance(raw_places, list):
            break

        for p in raw_places:
            pid = p.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                collected.append(p)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return collected


def normalize_place(p: Dict[str, Any], category: str, location: str) -> Dict[str, Any]:
    place_id = str(p.get("id") or "")
    display_name_obj = p.get("displayName")
    if isinstance(display_name_obj, dict):
        name = str(display_name_obj.get("text") or "")
    else:
        name = str(display_name_obj or "")

    address = str(p.get("formattedAddress") or "")
    locality = extract_locality(address, location)
    website = str(p.get("websiteUri")) if p.get("websiteUri") else None
    rating = float(p.get("rating", 0.0) or 0.0)
    review_count = int(p.get("userRatingCount", 0) or 0)

    return {
        "place_id": place_id,
        "name": name,
        "category": category,
        "address": address,
        "locality": locality,
        "website": website,
        "rating": rating,
        "review_count": review_count,
        "quality": {"rating_pct": 0, "volume_pct": 0, "score": 0},
        "site": {"performance": 0, "seo": 0, "findings": [], "audited": False, "no_website": (website is None)},
        "ai": {"queries": [], "mentions": 0, "total": 0, "visibility": 0},
        "competitors": [],
        "visibility_score": 0,
        "gold_score": 0,
        "recommended_service": "",
        "outreach": None,
        "scanned_at": datetime.now(timezone.utc).isoformat()
    }


def compute_quality_and_filter(businesses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Quality, computed in Python, never asked of the model.
    Drop anything under 20 reviews.
    Then across the remaining set:
    rating_pct = percentile rank of rating.
    volume_pct = percentile rank of log10(review_count + 1).
    quality.score = round(0.6*rating_pct + 0.4*volume_pct).
    Store all three.
    Filter: Keep the top 20 by quality.score.
    """
    # Drop anything under 20 reviews
    qualified = [b for b in businesses if b.get("review_count", 0) >= 20]
    if not qualified:
        return []

    ratings = [float(b.get("rating", 0.0)) for b in qualified]
    volumes = [math.log10(b.get("review_count", 0) + 1) for b in qualified]

    rating_pcts = compute_percentile_ranks(ratings)
    volume_pcts = compute_percentile_ranks(volumes)

    for i, b in enumerate(qualified):
        rp = rating_pcts[i]
        vp = volume_pcts[i]
        score = round(0.6 * rp + 0.4 * vp)
        b["quality"] = {
            "rating_pct": rp,
            "volume_pct": vp,
            "score": score
        }

    # Sort descending by quality.score
    qualified.sort(key=lambda x: (x["quality"]["score"], x["review_count"], x["rating"]), reverse=True)
    # Keep top 20
    return qualified[:20]


async def run_discovery_pipeline(run_id: str, category: str, location: str, services: str):
    """Background task orchestrating discovering and qualifying stages."""
    try:
        # Phase 1: Discovering
        db_save_run(run_id, {
            "status": "running",
            "stage": "discovering",
            "stage_status": {
                "discovering": "running",
                "qualifying": "pending",
                "auditing": "not_implemented",
                "testing": "not_implemented",
                "comparing": "not_implemented",
                "complete": "not_implemented"
            },
            "stage_counts": {}
        })
        
        raw_places = await fetch_places_paginated(category, location, max_pages=3)
        normalized = [normalize_place(p, category, location) for p in raw_places]
        candidates_count = len(normalized)

        # Update discovering count and set stage to qualifying
        db_save_run(run_id, {
            "status": "running",
            "stage": "qualifying",
            "stage_status": {
                "discovering": "done",
                "qualifying": "running",
                "auditing": "not_implemented",
                "testing": "not_implemented",
                "comparing": "not_implemented",
                "complete": "not_implemented"
            },
            "stage_counts": {
                "discovering": candidates_count
            }
        })

        # Phase 2: Qualifying
        top_20 = compute_quality_and_filter(normalized)
        qualified_count = len(top_20)

        # Persist qualified and candidate businesses to database
        for b in top_20:
            db_save_business(b["place_id"], b)
        for b in normalized:
            if b["place_id"] not in [t["place_id"] for t in top_20]:
                db_save_business(b["place_id"], b)

        # Auditing, AI discoverability, competitor comparison and gold scoring are not_implemented
        # Finish run after qualifying writes data
        db_save_run(run_id, {
            "status": "complete",
            "stage": "complete",
            "stage_status": {
                "discovering": "done",
                "qualifying": "done",
                "auditing": "not_implemented",
                "testing": "not_implemented",
                "comparing": "not_implemented",
                "complete": "not_implemented"
            },
            "stage_counts": {
                "discovering": candidates_count,
                "qualifying": qualified_count
            },
            "results": top_20,
            "finished_at": datetime.now(timezone.utc).isoformat()
        })

    except Exception as err:
        print(f"Error in discovery pipeline run {run_id}: {err}")
        db_save_run(run_id, {
            "status": "error",
            "stage": "error",
            "error": str(err),
            "finished_at": datetime.now(timezone.utc).isoformat()
        })


# API Models
class DiscoverInput(BaseModel):
    category: Optional[str] = "Restaurants and cafés"
    location: Optional[str] = "London"
    services: Optional[str] = "SEO, websites and AI visibility"


class SeedInput(BaseModel):
    location: Optional[str] = "London"


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """GET /api/health returns status, authMode: vertex, and the actual model constant."""
    return {
        "status": "ok",
        "authMode": "vertex",
        "model": MODEL
    }


@app.post("/api/discover")
async def post_discover(request: Request) -> Dict[str, Any]:
    """
    POST /api/discover creates the run document, returns {"run_id":"..."} immediately,
    and continues in an asynchronous background task.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    category = str(body.get("category") or "Restaurants and cafés").strip()
    location = str(body.get("location") or "London").strip()
    services = str(body.get("services") or "SEO, websites and AI visibility").strip()

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    run_doc: Dict[str, Any] = {
        "run_id": run_id,
        "category": category,
        "location": location,
        "services": services,
        "status": "running",
        "stage": "discovering",
        "stage_status": {
            "discovering": "running",
            "qualifying": "pending",
            "auditing": "not_implemented",
            "testing": "not_implemented",
            "comparing": "not_implemented",
            "complete": "not_implemented"
        },
        "stage_counts": {},
        "unmatched_names": [],
        "started_at": now_iso,
        "finished_at": None,
        "results": []
    }

    db_save_run(run_id, run_doc)
    # Run async background task non-blockingly
    asyncio.create_task(run_discovery_pipeline(run_id, category, location, services))

    return {"run_id": run_id}


@app.get("/api/run/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """GET /api/run/{run_id} returns the run document."""
    run_data = db_get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_data


@app.post("/api/seed")
async def post_seed(request: Request) -> Dict[str, Any]:
    """
    POST /api/seed runs discovery only, no audit and no Gemini, across all fifteen categories
    for a location, writing all of it to businesses.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    location = str(body.get("location") or "London").strip()

    async def seed_task(loc: str):
        for cat in ALL_CATEGORIES:
            try:
                raw_places = await fetch_places_paginated(cat, loc, max_pages=1)
                for p in raw_places:
                    b = normalize_place(p, cat, loc)
                    db_save_business(b["place_id"], b)
            except Exception as e:
                print(f"Error seeding category {cat}: {e}")

    asyncio.create_task(seed_task(location))
    return {"status": "seeding_started", "location": location, "categories": len(ALL_CATEGORIES)}


@app.get("/api/index/stats")
def get_index_stats() -> Dict[str, Any]:
    """GET /api/index/stats returns total businesses and distinct category count."""
    return db_get_index_stats()


@app.get("/")
def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Goldmine Service Running</h1>")


if __name__ == "__main__":
    import uvicorn
    # Listen on the port from the PORT environment variable, defaulting to 8080. Never hardcode 3000.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
