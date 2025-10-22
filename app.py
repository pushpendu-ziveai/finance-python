from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import asyncio
import aiofiles
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import json
import numpy as np
import pickle
import hashlib

# Import our custom modules
from libs.extract_pdf import extract_pdf
from libs.parse_transactions import parse_transactions
from libs.query_agent import query_agent
from libs.enhanced_pdf_extractor import EnhancedPDFExtractor, process_pdf_with_enhanced_extraction
from libs.enhanced_query_agent import EnhancedQueryAgent, query_enhanced_document

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Finance PDF Analysis API",
    description="Analyze PDF bank statements using AWS Textract and Bedrock",
    version="2.0.0"
)

# Add session middleware with more robust configuration
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("FASTAPI_SECRET_KEY", "finance-app-secret-key-python"),
    max_age=7200,  # 2 hours
    same_site="lax"
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create uploads and sessions directories if they don't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Supported file types
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file type and size"""
    if not file.filename:
        return False
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to disk and return the file path"""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem # type: ignore
    file_ext = Path(file.filename).suffix # type: ignore
    filename = f"{original_name}_{timestamp}{file_ext}"
    
    file_path = UPLOAD_DIR / filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return str(file_path)

async def upload_to_s3(file_path: str, s3_key: str, content_type: str) -> None:
    """Upload file to S3 bucket"""
    try:
        with open(file_path, 'rb') as file_data:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type
            )
        logger.info(f"File uploaded to S3: {S3_BUCKET_NAME}/{s3_key}")
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

def cleanup_file(file_path: str) -> None:
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

def get_session_id(request: Request) -> str:
    """Generate a session ID from request IP and user agent"""
    user_agent = request.headers.get("user-agent", "")
    client_ip = request.client.host if request.client else "unknown"
    session_string = f"{client_ip}_{user_agent}"
    return hashlib.md5(session_string.encode()).hexdigest()

def save_session_data(session_id: str, data: Dict[str, Any]) -> None:
    """Save session data to file"""
    try:
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        with open(session_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Session data saved: {session_id}")
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")

def load_session_data(session_id: str) -> Dict[str, Any]:
    """Load session data from file"""
    try:
        session_file = SESSIONS_DIR / f"{session_id}.pkl"
        if session_file.exists():
            with open(session_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Session data loaded: {session_id}")
            return data
        return {}
    except Exception as e:
        logger.error(f"Failed to load session data: {e}")
        return {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/query", response_class=HTMLResponse)
async def query_page(request: Request):
    """Query page with transaction data"""
    transactions = request.session.get("transactions", [])
    extraction_info = request.session.get("extraction_info", {})
    
    # If no session data, try to restore from persistent storage
    if not transactions:
        session_id = get_session_id(request)
        saved_data = load_session_data(session_id)
        if saved_data:
            transactions = saved_data.get("transactions", [])
            extraction_info = saved_data.get("extraction_info", {})
            
            # Restore to session
            request.session["transactions"] = transactions
            request.session["extraction_info"] = extraction_info
            logger.info(f"Restored session data - Transactions count: {len(transactions)}")
    
    # Debug logging
    logger.info(f"Query page accessed - Transactions count: {len(transactions)}")
    if extraction_info:
        logger.info(f"Extraction info: {extraction_info}")
    
    return templates.TemplateResponse("query.html", {
        "request": request,
        "transactions": transactions,
        "extraction_info": extraction_info
    })



@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    
    # Validate file
    if not validate_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload PDF, JPG, JPEG, or PNG files only."
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail="File size exceeds 10MB limit."
        )
    
    local_file_path = None
    
    try:
        # Save uploaded file locally
        local_file_path = await save_uploaded_file(file)
        logger.info(f"File saved locally: {local_file_path}")
        
        # Generate S3 key
        s3_key = f"uploads/{Path(local_file_path).name}"
        content_type = file.content_type or "application/pdf"
        
        # Upload to S3
        await upload_to_s3(local_file_path, s3_key, content_type)
        
        # Use enhanced extraction system with fallback
        logger.info("🚀 Starting enhanced PDF extraction...")
        
        extraction_method = "unknown"
        transactions = []
        extraction_info = {}
        
        # Validate S3 bucket configuration
        if not S3_BUCKET_NAME:
            raise HTTPException(status_code=500, detail="S3 bucket not configured")
        
        try:
            # Try enhanced multilayer extraction first
            enhanced_result = await process_pdf_with_enhanced_extraction(S3_BUCKET_NAME, s3_key)
            
            if enhanced_result and "error" not in enhanced_result:
                # Enhanced extraction succeeded
                transactions = enhanced_result["transactions"]
                extraction_method = "enhanced_multilayer_hybrid"
                extraction_info = {
                    "filename": file.filename,
                    "s3_key": s3_key,
                    "document_structure": enhanced_result["document_structure"],
                    "extraction_method": extraction_method,
                    "semantic_queries": enhanced_result.get("semantic_queries", {}),
                    "total_elements": enhanced_result["document_structure"]["total_elements"]
                }
                logger.info(f"✅ Enhanced extraction successful: {len(transactions)} transactions, {extraction_info['total_elements']} elements")
            else:
                raise Exception("Enhanced extraction returned error")
                
        except Exception as enhanced_error:
            logger.warning(f"Enhanced extraction failed: {enhanced_error}")
            logger.info("📋 Falling back to standard Textract extraction...")
            
            # Fallback to standard Textract extraction
            textract_data = await extract_pdf(S3_BUCKET_NAME, s3_key)
            
            if textract_data and textract_data.get('Blocks'):
                logger.info(f"Standard extraction - Total blocks: {len(textract_data['Blocks'])}")
                
                # Parse transactions from standard extraction
                transactions = parse_transactions(textract_data)
                extraction_method = "aws_textract_fallback"
                extraction_info = {
                    "filename": file.filename,
                    "s3_key": s3_key,
                    "total_blocks": len(textract_data.get('Blocks', [])),
                    "extraction_method": extraction_method
                }
                logger.info(f"✅ Standard extraction completed: {len(transactions)} transactions")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Both enhanced and standard extraction failed. Please try a different file."
                )
        
        # Check if we got any transactions
        if not transactions:
            logger.warning("No transactions found in the document")
            extraction_info["warning"] = "No transactions detected in the document"
        
        # Store in session
        request.session["transactions"] = transactions
        extraction_info.update({
            "total_transactions": len(transactions),
            "extracted_at": datetime.now().isoformat()
        })
        request.session["extraction_info"] = extraction_info
        
        # Persist session data to file
        session_id = get_session_id(request)
        save_session_data(session_id, {
            "transactions": transactions,
            "extraction_info": extraction_info
        })
        
        logger.info(f"Successfully processed {len(transactions)} transactions using {extraction_method}")
        
        # Create enhanced success message
        if extraction_method == "enhanced_multilayer_hybrid":
            message = f"🚀 Enhanced extraction completed! Found {len(transactions)} transactions with advanced semantic understanding."
            if extraction_info.get('total_elements'):
                message += f" Analyzed {extraction_info['total_elements']} document elements."
        else:
            message = f"✅ Successfully extracted {len(transactions)} transactions from your PDF using standard method."
        
        return JSONResponse({
            "success": True,
            "message": message,
            "transaction_count": len(transactions),
            "extraction_method": extraction_method,
            "enhanced_features": extraction_method == "enhanced_multilayer_hybrid",
            "redirect": "/query"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up local file
        if local_file_path:
            cleanup_file(local_file_path)

@app.post("/query")
async def handle_query(request: Request, query: str = Form(...)):
    """Handle AI query about transactions with enhanced capabilities"""
    
    transactions = request.session.get("transactions", [])
    extraction_info = request.session.get("extraction_info", {})
    
    # If no session data, try to restore from persistent storage
    if not transactions:
        session_id = get_session_id(request)
        saved_data = load_session_data(session_id)
        if saved_data:
            transactions = saved_data.get("transactions", [])
            extraction_info = saved_data.get("extraction_info", {})
            
            # Restore to session
            request.session["transactions"] = transactions
            request.session["extraction_info"] = extraction_info
            logger.info(f"Restored session data for query - Transactions count: {len(transactions)}")
    
    if not transactions:
        raise HTTPException(
            status_code=400, 
            detail="No transactions found. Please upload a PDF first."
        )
    
    try:
        logger.info(f"🔍 Processing query: '{query}' (Method: {extraction_info.get('extraction_method', 'unknown')})")
        
        # Check if we have enhanced extraction data available
        extraction_method = extraction_info.get("extraction_method", "")
        
        if extraction_method == "enhanced_multilayer_hybrid":
            # Use enhanced query capabilities
            logger.info("Using enhanced semantic query processing...")
            
            # Create a simplified enhanced query agent
            agent = EnhancedQueryAgent()
            
            # Perform semantic search on transactions
            query_embedding = agent.embedding_model.encode([query])[0]
            relevant_transactions = []
            
            for transaction in transactions:
                description = transaction.get('description', '')
                if description:
                    desc_embedding = agent.embedding_model.encode([description])[0]
                    similarity = np.dot(query_embedding, desc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(desc_embedding)
                    )
                    if similarity > 0.3:  # Similarity threshold
                        transaction['relevance_score'] = float(similarity)
                        relevant_transactions.append(transaction)
            
            # Sort by relevance
            relevant_transactions.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Generate enhanced response
            context_info = f"""
            Document Analysis:
            - Extraction Method: Enhanced Multilayer Hybrid
            - Total Elements: {extraction_info.get('total_elements', 'N/A')}
            - Document Structure: {extraction_info.get('document_structure', {}).get('document_type', 'Unknown')}
            - Pages: {extraction_info.get('document_structure', {}).get('pages', 'N/A')}
            
            Query: {query}
            
            Relevant Transactions ({len(relevant_transactions)} found):
            """
            
            for i, t in enumerate(relevant_transactions[:10], 1):
                score = t.get('relevance_score', 0) * 100
                context_info += f"{i}. {t.get('date', 'N/A')} - {t.get('description', 'N/A')} - {t.get('amount', 'N/A')} (Relevance: {score:.1f}%)\n"
            
            response = await agent._call_claude(context_info + "\nPlease provide a comprehensive analysis based on this financial data.")
            
            query_result = {
                "success": True,
                "query": query,
                "response": response,
                "transaction_count": len(transactions),
                "relevant_transactions": len(relevant_transactions),
                "extraction_method": "enhanced_semantic_search",
                "enhanced_features": {
                    "semantic_search": True,
                    "relevance_scoring": True,
                    "document_structure_analysis": True
                }
            }
            
        else:
            # Use standard query processing
            logger.info("Using standard query processing...")
            response = await query_agent(query, transactions)
            
            query_result = {
                "success": True,
                "query": query,
                "response": response,
                "transaction_count": len(transactions),
                "extraction_method": "standard_query"
            }
        
        return JSONResponse(query_result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/transactions")
async def get_transactions(request: Request):
    """Get stored transactions"""
    transactions = request.session.get("transactions", [])
    extraction_info = request.session.get("extraction_info", {})
    
    return JSONResponse({
        "transactions": transactions,
        "info": extraction_info,
        "count": len(transactions)
    })

@app.delete("/api/transactions")
async def clear_transactions(request: Request):
    """Clear stored transactions"""
    request.session.pop("transactions", None)
    request.session.pop("extraction_info", None)
    
    return JSONResponse({
        "success": True,
        "message": "Transactions cleared"
    })



@app.get("/debug/session")
async def debug_session(request: Request):
    """Debug session data"""
    transactions = request.session.get("transactions", [])
    extraction_info = request.session.get("extraction_info", {})
    
    logger.info(f"Debug session - Transactions: {len(transactions)}, Keys: {list(request.session.keys())}")
    
    return JSONResponse({
        "transactions_count": len(transactions),
        "extraction_info": extraction_info,
        "session_keys": list(request.session.keys()),
        "sample_transaction": transactions[0] if transactions else None
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "finance-pdf-analysis",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 3000
    port = int(os.getenv("PORT", 3000))
    
    logger.info(f"Starting FastAPI server on http://localhost:{port}")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )