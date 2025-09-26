"""
FastAPI Backend for Document Layout Analysis
===========================================

A RESTful API for document layout analysis with:
- Document upload endpoints
- Analysis processing
- Results retrieval
- Batch processing
- WebSocket support for real-time updates

Author: AI Assistant
Date: 2024
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import asyncio
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Import our analyzer
from modern_layout_analyzer import ModernDocumentLayoutAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Layout Analysis API",
    description="Advanced document layout analysis with multiple ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = ModernDocumentLayoutAnalyzer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Layout Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "analyze": "/analyze",
            "history": "/history",
            "models": "/models",
            "websocket": "/ws"
        }
    }

@app.get("/models")
async def get_available_models():
    """Get list of available analysis models"""
    return {
        "available_models": list(analyzer.models.keys()),
        "default_model": "layoutparser",
        "model_info": {
            "layoutparser": "LayoutParser with Detectron2 (PubLayNet)",
            "yolo": "YOLOv8 for object detection"
        }
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "temp_path": tmp_path,
            "size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    model: str = "layoutparser",
    extract_text: bool = True,
    save_results: bool = True
):
    """Analyze uploaded document"""
    try:
        # Validate model
        if model not in analyzer.models:
            raise HTTPException(status_code=400, detail=f"Model {model} not available")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Perform analysis
        results = analyzer.analyze_document(
            tmp_path,
            model_name=model,
            extract_text=extract_text,
            save_results=save_results
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    model: str = "layoutparser",
    extract_text: bool = True,
    save_results: bool = True
):
    """Analyze multiple documents in batch"""
    try:
        batch_results = []
        
        for file in files:
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name
                
                # Perform analysis
                result = analyzer.analyze_document(
                    tmp_path,
                    model_name=model,
                    extract_text=extract_text,
                    save_results=save_results
                )
                
                batch_results.append(result)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                batch_results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "batch_results": batch_results,
            "total_files": len(files),
            "successful": len([r for r in batch_results if r.get("success", True)])
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/history")
async def get_analysis_history():
    """Get analysis history from database"""
    try:
        history_df = analyzer.get_analysis_history()
        return {
            "history": history_df.to_dict('records'),
            "total_analyses": len(history_df)
        }
    except Exception as e:
        logger.error(f"Failed to get history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/history/{analysis_id}")
async def get_analysis_by_id(analysis_id: int):
    """Get specific analysis by ID"""
    try:
        analysis = analyzer.session.query(analyzer.DocumentAnalysis).filter_by(id=analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {
            "id": analysis.id,
            "filename": analysis.filename,
            "model_used": analysis.model_used,
            "analysis_date": analysis.analysis_date.isoformat(),
            "layout_data": analysis.layout_data,
            "ocr_text": analysis.ocr_text,
            "confidence_scores": analysis.confidence_scores,
            "processing_time": analysis.processing_time
        }
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@app.delete("/history")
async def clear_analysis_history():
    """Clear all analysis history"""
    try:
        analyzer.session.query(analyzer.DocumentAnalysis).delete()
        analyzer.session.commit()
        return {"message": "Analysis history cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get("/export/{analysis_id}")
async def export_analysis(analysis_id: int, format: str = "json"):
    """Export analysis results in specified format"""
    try:
        analysis = analyzer.session.query(analyzer.DocumentAnalysis).filter_by(id=analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Prepare results
        results = {
            "id": analysis.id,
            "filename": analysis.filename,
            "model_used": analysis.model_used,
            "analysis_date": analysis.analysis_date.isoformat(),
            "layout_blocks": analysis.layout_data,
            "ocr_text": analysis.ocr_text,
            "confidence_scores": analysis.confidence_scores,
            "processing_time": analysis.processing_time
        }
        
        if format == "json":
            return JSONResponse(content=results)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(results["layout_blocks"])
            csv_content = df.to_csv(index=False)
            return JSONResponse(content={"csv": csv_content})
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Failed to export analysis {analysis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export analysis: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Echo back the message
            await manager.send_personal_message(f"Echo: {data}", websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(analyzer.models),
        "database_connected": True
    }

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        history_df = analyzer.get_analysis_history()
        
        if history_df.empty:
            return {
                "total_analyses": 0,
                "average_processing_time": 0,
                "most_used_model": "N/A",
                "total_text_extracted": 0
            }
        
        return {
            "total_analyses": len(history_df),
            "average_processing_time": float(history_df['processing_time'].mean()),
            "most_used_model": history_df['model_used'].mode().iloc[0] if not history_df.empty else "N/A",
            "total_text_extracted": int(history_df['ocr_text'].str.len().sum()) if 'ocr_text' in history_df.columns else 0
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
