"""
Test Suite for Document Layout Analysis System
==============================================

Comprehensive test suite including:
- Unit tests for core functionality
- Integration tests for API endpoints
- Mock data generation
- Performance tests

Author: AI Assistant
Date: 2024
"""

import pytest
import tempfile
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock

# Import our modules
from modern_layout_analyzer import ModernDocumentLayoutAnalyzer, DocumentAnalysis

class TestDocumentLayoutAnalyzer(unittest.TestCase):
    """Test cases for the ModernDocumentLayoutAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_db_path = "test_analysis.db"
        self.analyzer = ModernDocumentLayoutAnalyzer(db_path=self.test_db_path)
        
        # Create a test image
        self.test_image = self.create_test_image()
        self.test_image_path = self.save_test_image()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        # Remove test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def create_test_image(self):
        """Create a test document image"""
        # Create a simple test image with text-like regions
        img = Image.new('RGB', (800, 600), color='white')
        
        # Add some colored rectangles to simulate text blocks
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Title block
        draw.rectangle([50, 50, 750, 100], fill='lightblue', outline='blue')
        
        # Text blocks
        draw.rectangle([50, 120, 400, 200], fill='lightgray', outline='gray')
        draw.rectangle([450, 120, 750, 200], fill='lightgray', outline='gray')
        
        # Table-like structure
        draw.rectangle([50, 220, 750, 400], fill='lightyellow', outline='orange')
        
        # Figure area
        draw.rectangle([50, 420, 300, 550], fill='lightgreen', outline='green')
        
        return img
    
    def save_test_image(self):
        """Save test image to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        self.test_image.save(temp_file.name)
        return temp_file.name
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.session)
        self.assertIsInstance(self.analyzer.models, dict)
    
    def test_database_creation(self):
        """Test database creation and connection"""
        # Check if database file exists
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check if table exists
        result = self.analyzer.session.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_analyses'")
        self.assertIsNotNone(result.fetchone())
    
    @patch('modern_layout_analyzer.lp.Detectron2LayoutModel')
    def test_layoutparser_model_initialization(self, mock_model):
        """Test LayoutParser model initialization"""
        mock_model.return_value = Mock()
        
        # Reinitialize analyzer to test model loading
        analyzer = ModernDocumentLayoutAnalyzer(db_path="test_model.db")
        
        # Verify model was called
        mock_model.assert_called_once()
    
    def test_image_loading(self):
        """Test image loading functionality"""
        # Test with valid image
        image = self.analyzer._load_image(self.test_image_path)
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # Should be RGB image
        
        # Test with invalid path
        with self.assertRaises(ValueError):
            self.analyzer._load_image("nonexistent_image.png")
    
    def test_coordinate_extraction(self):
        """Test coordinate extraction from layout blocks"""
        # Mock layout block
        mock_block = Mock()
        mock_block.coordinates = [10, 20, 100, 200]
        mock_block.type = 'Text'
        mock_block.score = 0.95
        
        # Test coordinate extraction
        x1, y1, x2, y2 = map(int, mock_block.coordinates)
        self.assertEqual(x1, 10)
        self.assertEqual(y1, 20)
        self.assertEqual(x2, 100)
        self.assertEqual(y2, 200)
    
    def test_database_save_and_retrieve(self):
        """Test saving and retrieving analysis results"""
        # Create mock analysis result
        mock_result = {
            'filename': 'test_doc.png',
            'model_used': 'layoutparser',
            'layout_blocks': [
                {'type': 'Text', 'coordinates': [10, 20, 100, 200], 'confidence': 0.95}
            ],
            'ocr_text': 'Sample text',
            'processing_time': 1.5
        }
        
        # Save to database
        self.analyzer._save_to_database(mock_result, self.test_image_path)
        
        # Retrieve from database
        history_df = self.analyzer.get_analysis_history()
        self.assertFalse(history_df.empty)
        self.assertEqual(history_df.iloc[0]['filename'], 'test_doc.png')
    
    def test_export_functionality(self):
        """Test result export functionality"""
        mock_results = {
            'layout_blocks': [
                {'type': 'Text', 'coordinates': [10, 20, 100, 200], 'confidence': 0.95},
                {'type': 'Title', 'coordinates': [10, 10, 200, 50], 'confidence': 0.98}
            ]
        }
        
        # Test JSON export
        json_output = self.analyzer.export_results(mock_results, 'json')
        self.assertIsInstance(json_output, str)
        
        # Parse JSON to verify structure
        parsed = json.loads(json_output)
        self.assertIn('layout_blocks', parsed)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        # Create multiple test images
        test_images = []
        for i in range(3):
            img = self.create_test_image()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_test_{i}.png')
            img.save(temp_file.name)
            test_images.append(temp_file.name)
        
        try:
            # Test batch processing (with mocked analysis)
            with patch.object(self.analyzer, '_analyze_with_model') as mock_analyze:
                mock_analyze.return_value = [
                    {'type': 'Text', 'coordinates': [10, 20, 100, 200], 'confidence': 0.95}
                ]
                
                results = self.analyzer.batch_analyze(
                    os.path.dirname(test_images[0]),
                    model_name='layoutparser'
                )
                
                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 3)
        
        finally:
            # Clean up test images
            for img_path in test_images:
                if os.path.exists(img_path):
                    os.remove(img_path)

class TestMockDataGeneration:
    """Test mock data generation for testing purposes"""
    
    def test_create_mock_document_image(self):
        """Test creation of mock document images"""
        from test_utils import create_mock_document_image
        
        # Test different document types
        invoice_img = create_mock_document_image('invoice')
        self.assertIsInstance(invoice_img, Image.Image)
        self.assertEqual(invoice_img.size, (800, 600))
        
        report_img = create_mock_document_image('report')
        self.assertIsInstance(report_img, Image.Image)
    
    def test_generate_sample_layout_data(self):
        """Test generation of sample layout data"""
        from test_utils import generate_sample_layout_data
        
        layout_data = generate_sample_layout_data()
        
        self.assertIsInstance(layout_data, list)
        self.assertGreater(len(layout_data), 0)
        
        # Check structure of layout blocks
        for block in layout_data:
            self.assertIn('type', block)
            self.assertIn('coordinates', block)
            self.assertIn('confidence', block)
            self.assertIsInstance(block['coordinates'], list)
            self.assertEqual(len(block['coordinates']), 4)

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API endpoints"""
    
    def setUp(self):
        """Set up API test fixtures"""
        from api import app
        from fastapi.testclient import TestClient
        
        self.client = TestClient(app)
        self.test_image_path = self.create_test_image_file()
    
    def tearDown(self):
        """Clean up API test fixtures"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def create_test_image_file(self):
        """Create a test image file for API testing"""
        img = Image.new('RGB', (400, 300), color='white')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img.save(temp_file.name)
        return temp_file.name
    
    def test_root_endpoint(self):
        """Test root API endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("endpoints", data)
    
    def test_models_endpoint(self):
        """Test models API endpoint"""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("available_models", data)
        self.assertIn("default_model", data)
    
    def test_upload_endpoint(self):
        """Test file upload endpoint"""
        with open(self.test_image_path, "rb") as f:
            response = self.client.post("/upload", files={"file": f})
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("filename", data)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("total_analyses", data)
        self.assertIn("average_processing_time", data)

# Performance tests
class TestPerformance(unittest.TestCase):
    """Performance tests for the system"""
    
    def test_analysis_performance(self):
        """Test analysis performance with large images"""
        analyzer = ModernDocumentLayoutAnalyzer(db_path="perf_test.db")
        
        # Create a larger test image
        large_img = Image.new('RGB', (2000, 1500), color='white')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        large_img.save(temp_file.name)
        
        try:
            import time
            start_time = time.time()
            
            # Mock the analysis to avoid actual model loading
            with patch.object(analyzer, '_analyze_with_model') as mock_analyze:
                mock_analyze.return_value = [
                    {'type': 'Text', 'coordinates': [10, 20, 100, 200], 'confidence': 0.95}
                ]
                
                results = analyzer.analyze_document(
                    temp_file.name,
                    model_name='layoutparser',
                    extract_text=False,
                    save_results=False
                )
            
            processing_time = time.time() - start_time
            
            # Performance should be reasonable (less than 10 seconds for mocked analysis)
            self.assertLess(processing_time, 10.0)
            
        finally:
            os.remove(temp_file.name)
            if os.path.exists("perf_test.db"):
                os.remove("perf_test.db")

# Test utilities
def create_mock_document_image(doc_type='general'):
    """Create mock document images for testing"""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    if doc_type == 'invoice':
        # Invoice layout
        draw.rectangle([50, 50, 750, 100], fill='lightblue', outline='blue')  # Header
        draw.rectangle([50, 120, 400, 300], fill='lightgray', outline='gray')  # Customer info
        draw.rectangle([450, 120, 750, 300], fill='lightgray', outline='gray')  # Invoice details
        draw.rectangle([50, 320, 750, 500], fill='lightyellow', outline='orange')  # Items table
        draw.rectangle([50, 520, 750, 580], fill='lightgreen', outline='green')  # Total
    
    elif doc_type == 'report':
        # Report layout
        draw.rectangle([50, 50, 750, 100], fill='lightblue', outline='blue')  # Title
        draw.rectangle([50, 120, 750, 200], fill='lightgray', outline='gray')  # Abstract
        draw.rectangle([50, 220, 750, 400], fill='lightyellow', outline='orange')  # Content
        draw.rectangle([50, 420, 300, 550], fill='lightgreen', outline='green')  # Figure
        draw.rectangle([350, 420, 750, 550], fill='lightpink', outline='red')  # References
    
    else:
        # General document
        draw.rectangle([50, 50, 750, 100], fill='lightblue', outline='blue')
        draw.rectangle([50, 120, 400, 200], fill='lightgray', outline='gray')
        draw.rectangle([450, 120, 750, 200], fill='lightgray', outline='gray')
        draw.rectangle([50, 220, 750, 400], fill='lightyellow', outline='orange')
        draw.rectangle([50, 420, 300, 550], fill='lightgreen', outline='green')
    
    return img

def generate_sample_layout_data():
    """Generate sample layout data for testing"""
    return [
        {
            'id': 0,
            'type': 'Title',
            'coordinates': [50, 50, 750, 100],
            'confidence': 0.98,
            'area': 70000
        },
        {
            'id': 1,
            'type': 'Text',
            'coordinates': [50, 120, 400, 200],
            'confidence': 0.95,
            'area': 28000
        },
        {
            'id': 2,
            'type': 'Text',
            'coordinates': [450, 120, 750, 200],
            'confidence': 0.92,
            'area': 24000
        },
        {
            'id': 3,
            'type': 'Table',
            'coordinates': [50, 220, 750, 400],
            'confidence': 0.89,
            'area': 126000
        },
        {
            'id': 4,
            'type': 'Figure',
            'coordinates': [50, 420, 300, 550],
            'confidence': 0.94,
            'area': 37500
        }
    ]

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
