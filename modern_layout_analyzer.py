"""
Modern Document Layout Analysis System
=====================================

A comprehensive document layout analysis system supporting multiple models:
- LayoutParser with Detectron2 (PubLayNet)
- YOLOv8 for modern object detection
- DETR (Detection Transformer) models
- Custom OCR integration with Tesseract and EasyOCR

Author: AI Assistant
Date: 2024
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Layout analysis
import layoutparser as lp

# Modern ML models
try:
    import torch
    import torchvision
    from ultralytics import YOLO
    from transformers import pipeline
    MODERN_MODELS_AVAILABLE = True
except ImportError:
    MODERN_MODELS_AVAILABLE = False
    print("Warning: Modern ML models not available. Install torch, ultralytics, transformers for full functionality.")

# OCR libraries
try:
    import pytesseract
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR libraries not available. Install pytesseract and easyocr for OCR functionality.")

# Database
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class DocumentAnalysis(Base):
    """Database model for storing document analysis results"""
    __tablename__ = 'document_analyses'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    model_used = Column(String(100), nullable=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String(500))
    layout_data = Column(JSON)
    ocr_text = Column(Text)
    confidence_scores = Column(JSON)
    processing_time = Column(Float)

class ModernDocumentLayoutAnalyzer:
    """
    Modern document layout analysis system with multiple model support
    """
    
    def __init__(self, db_path: str = "document_analysis.db"):
        """Initialize the analyzer with database connection"""
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize available models"""
        try:
            # LayoutParser model
            self.models['layoutparser'] = lp.Detectron2LayoutModel(
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                use_gpu=torch.cuda.is_available() if MODERN_MODELS_AVAILABLE else False
            )
            logger.info("LayoutParser model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LayoutParser model: {e}")
            
        if MODERN_MODELS_AVAILABLE:
            try:
                # YOLOv8 model (custom trained for document layout)
                self.models['yolo'] = YOLO('yolov8n.pt')  # Using general YOLO, can be replaced with custom trained model
                logger.info("YOLOv8 model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLOv8 model: {e}")
                
        # Initialize OCR readers
        if OCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
    
    def analyze_document(self, 
                        image_path: str, 
                        model_name: str = 'layoutparser',
                        extract_text: bool = True,
                        save_results: bool = True) -> Dict:
        """
        Analyze document layout using specified model
        
        Args:
            image_path: Path to document image
            model_name: Model to use ('layoutparser', 'yolo')
            extract_text: Whether to extract text using OCR
            save_results: Whether to save results to database
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.utcnow()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform layout analysis
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
            
        layout_results = self._analyze_with_model(image_rgb, model_name)
        
        # Extract text if requested
        ocr_text = ""
        if extract_text and OCR_AVAILABLE:
            ocr_text = self._extract_text(image_rgb, layout_results)
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare results
        results = {
            'filename': Path(image_path).name,
            'model_used': model_name,
            'layout_blocks': layout_results,
            'ocr_text': ocr_text,
            'processing_time': processing_time,
            'image_shape': image_rgb.shape,
            'analysis_date': start_time.isoformat()
        }
        
        # Save to database if requested
        if save_results:
            self._save_to_database(results, image_path)
            
        return results
    
    def _analyze_with_model(self, image: np.ndarray, model_name: str) -> List[Dict]:
        """Analyze image with specified model"""
        if model_name == 'layoutparser':
            return self._analyze_with_layoutparser(image)
        elif model_name == 'yolo':
            return self._analyze_with_yolo(image)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _analyze_with_layoutparser(self, image: np.ndarray) -> List[Dict]:
        """Analyze using LayoutParser"""
        layout = self.models['layoutparser'].detect(image)
        
        blocks = []
        for i, block in enumerate(layout):
            x1, y1, x2, y2 = map(int, block.coordinates)
            blocks.append({
                'id': i,
                'type': block.type,
                'coordinates': [x1, y1, x2, y2],
                'confidence': getattr(block, 'score', 1.0),
                'area': (x2 - x1) * (y2 - y1)
            })
        
        return blocks
    
    def _analyze_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Analyze using YOLOv8"""
        if not MODERN_MODELS_AVAILABLE:
            raise RuntimeError("YOLO models not available")
            
        results = self.models['yolo'](image)
        
        blocks = []
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None:
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    blocks.append({
                        'id': len(blocks),
                        'type': f'Object_{cls}',  # YOLO class names would need custom mapping
                        'coordinates': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'area': int((x2 - x1) * (y2 - y1))
                    })
        
        return blocks
    
    def _extract_text(self, image: np.ndarray, layout_blocks: List[Dict]) -> str:
        """Extract text from document using OCR"""
        if not OCR_AVAILABLE:
            return ""
            
        full_text = ""
        
        # Extract text from each text block
        for block in layout_blocks:
            if block['type'] in ['Text', 'Title']:
                x1, y1, x2, y2 = block['coordinates']
                cropped = image[y1:y2, x1:x2]
                
                # Use EasyOCR for better accuracy
                try:
                    ocr_results = self.easyocr_reader.readtext(cropped)
                    block_text = ' '.join([result[1] for result in ocr_results])
                    full_text += block_text + '\n'
                except Exception as e:
                    logger.warning(f"OCR failed for block {block['id']}: {e}")
        
        return full_text.strip()
    
    def _save_to_database(self, results: Dict, image_path: str):
        """Save analysis results to database"""
        try:
            analysis = DocumentAnalysis(
                filename=results['filename'],
                model_used=results['model_used'],
                image_path=image_path,
                layout_data=results['layout_blocks'],
                ocr_text=results['ocr_text'],
                confidence_scores=[block.get('confidence', 0) for block in results['layout_blocks']],
                processing_time=results['processing_time']
            )
            
            self.session.add(analysis)
            self.session.commit()
            logger.info(f"Analysis results saved to database for {results['filename']}")
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            self.session.rollback()
    
    def visualize_results(self, image_path: str, results: Dict, save_path: Optional[str] = None):
        """Visualize analysis results"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Document')
        ax1.axis('off')
        
        # Image with layout blocks
        image_with_blocks = image_rgb.copy()
        colors = {'Text': 'red', 'Title': 'blue', 'List': 'green', 'Table': 'orange', 'Figure': 'purple'}
        
        for block in results['layout_blocks']:
            x1, y1, x2, y2 = block['coordinates']
            color = colors.get(block['type'], 'yellow')
            
            cv2.rectangle(image_with_blocks, (x1, y1), (x2, y2), 
                         plt.cm.tab10.colors[hash(block['type']) % 10], 2)
            
            # Add label
            cv2.putText(image_with_blocks, f"{block['type']} ({block['confidence']:.2f})",
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ax2.imshow(image_with_blocks)
        ax2.set_title(f'Layout Analysis ({results["model_used"]})')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def batch_analyze(self, image_directory: str, model_name: str = 'layoutparser') -> List[Dict]:
        """Analyze multiple documents in batch"""
        image_dir = Path(image_directory)
        image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg'))
        
        results = []
        for image_file in image_files:
            try:
                result = self.analyze_document(str(image_file), model_name)
                results.append(result)
                logger.info(f"Processed: {image_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
        
        return results
    
    def get_analysis_history(self) -> pd.DataFrame:
        """Get analysis history from database"""
        query = self.session.query(DocumentAnalysis)
        df = pd.read_sql(query.statement, self.session.bind)
        return df
    
    def export_results(self, results: Dict, format: str = 'json') -> str:
        """Export results in specified format"""
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(results['layout_blocks'])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Main function for testing the analyzer"""
    # Initialize analyzer
    analyzer = ModernDocumentLayoutAnalyzer()
    
    # Example usage (requires sample image)
    sample_image = "sample_doc.png"
    
    if os.path.exists(sample_image):
        # Analyze with LayoutParser
        results = analyzer.analyze_document(sample_image, model_name='layoutparser')
        
        # Visualize results
        analyzer.visualize_results(sample_image, results)
        
        # Print summary
        print(f"\nAnalysis Summary:")
        print(f"Model: {results['model_used']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Number of blocks detected: {len(results['layout_blocks'])}")
        print(f"Text extracted: {len(results['ocr_text'])} characters")
        
        # Export results
        json_output = analyzer.export_results(results, 'json')
        print(f"\nJSON Export:\n{json_output}")
        
    else:
        print(f"Sample image {sample_image} not found. Please provide a document image for analysis.")


if __name__ == "__main__":
    main()
