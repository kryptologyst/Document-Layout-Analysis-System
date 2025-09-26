"""
Sample Data Generator for Document Layout Analysis
=================================================

Generates sample documents and mock data for testing and demonstration purposes.

Author: AI Assistant
Date: 2024
"""

import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_documents():
    """Create sample document images for testing"""
    
    # Create samples directory
    samples_dir = Path("sample_data")
    samples_dir.mkdir(exist_ok=True)
    
    # Document types to generate
    document_types = {
        'invoice': create_invoice_document,
        'report': create_report_document,
        'form': create_form_document,
        'article': create_article_document,
        'receipt': create_receipt_document
    }
    
    for doc_type, create_func in document_types.items():
        print(f"Creating {doc_type} document...")
        img = create_func()
        img.save(samples_dir / f"sample_{doc_type}.png")
    
    print(f"Sample documents created in {samples_dir}")

def create_invoice_document():
    """Create a sample invoice document"""
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    # Header section
    draw.rectangle([50, 50, 750, 120], fill='lightblue', outline='blue', width=2)
    draw.text((100, 80), "INVOICE", fill='darkblue')
    draw.text((600, 80), "INV-2024-001", fill='darkblue')
    
    # Company info
    draw.rectangle([50, 140, 400, 250], fill='lightgray', outline='gray')
    draw.text((70, 160), "From: ABC Company", fill='black')
    draw.text((70, 180), "123 Business St", fill='black')
    draw.text((70, 200), "City, State 12345", fill='black')
    
    # Customer info
    draw.rectangle([450, 140, 750, 250], fill='lightgray', outline='gray')
    draw.text((470, 160), "To: XYZ Corp", fill='black')
    draw.text((470, 180), "456 Customer Ave", fill='black')
    draw.text((470, 200), "City, State 67890", fill='black')
    
    # Invoice details
    draw.rectangle([50, 270, 750, 320], fill='lightyellow', outline='orange')
    draw.text((70, 290), "Invoice Date: 2024-01-15", fill='black')
    draw.text((400, 290), "Due Date: 2024-02-15", fill='black')
    
    # Items table header
    draw.rectangle([50, 340, 750, 380], fill='darkblue', outline='blue')
    draw.text((70, 355), "Description", fill='white')
    draw.text((300, 355), "Quantity", fill='white')
    draw.text((400, 355), "Price", fill='white')
    draw.text((500, 355), "Total", fill='white')
    
    # Items rows
    items = [
        ("Software License", "1", "$500.00", "$500.00"),
        ("Support Service", "12", "$50.00", "$600.00"),
        ("Consulting", "40", "$100.00", "$4000.00")
    ]
    
    y_pos = 400
    for item in items:
        draw.rectangle([50, y_pos, 750, y_pos + 30], fill='lightgray', outline='gray')
        draw.text((70, y_pos + 10), item[0], fill='black')
        draw.text((300, y_pos + 10), item[1], fill='black')
        draw.text((400, y_pos + 10), item[2], fill='black')
        draw.text((500, y_pos + 10), item[3], fill='black')
        y_pos += 35
    
    # Total section
    draw.rectangle([50, y_pos + 10, 750, y_pos + 80], fill='lightgreen', outline='green')
    draw.text((500, y_pos + 30), "Subtotal: $5100.00", fill='black')
    draw.text((500, y_pos + 50), "Tax: $510.00", fill='black')
    draw.text((500, y_pos + 70), "Total: $5610.00", fill='black')
    
    # Footer
    draw.rectangle([50, 900, 750, 950], fill='lightgray', outline='gray')
    draw.text((70, 920), "Thank you for your business!", fill='black')
    
    return img

def create_report_document():
    """Create a sample report document"""
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.rectangle([50, 50, 750, 120], fill='darkblue', outline='blue')
    draw.text((100, 80), "QUARTERLY BUSINESS REPORT", fill='white')
    
    # Abstract
    draw.rectangle([50, 140, 750, 220], fill='lightblue', outline='blue')
    draw.text((70, 160), "Abstract", fill='darkblue')
    draw.text((70, 180), "This report summarizes our Q4 performance...", fill='black')
    
    # Executive Summary
    draw.rectangle([50, 240, 750, 400], fill='lightgray', outline='gray')
    draw.text((70, 260), "Executive Summary", fill='black')
    draw.text((70, 280), "• Revenue increased by 15%", fill='black')
    draw.text((70, 300), "• Customer satisfaction improved", fill='black')
    draw.text((70, 320), "• New product launches successful", fill='black')
    draw.text((70, 340), "• Market expansion completed", fill='black')
    
    # Chart/Figure
    draw.rectangle([50, 420, 400, 600], fill='lightgreen', outline='green')
    draw.text((150, 500), "Revenue Chart", fill='black')
    
    # Data table
    draw.rectangle([450, 420, 750, 600], fill='lightyellow', outline='orange')
    draw.text((470, 440), "Key Metrics", fill='black')
    draw.text((470, 460), "Revenue: $2.5M", fill='black')
    draw.text((470, 480), "Profit: $500K", fill='black')
    draw.text((470, 500), "Customers: 1,250", fill='black')
    
    # Conclusion
    draw.rectangle([50, 620, 750, 800], fill='lightpink', outline='red')
    draw.text((70, 640), "Conclusion", fill='black')
    draw.text((70, 660), "The quarter showed strong growth...", fill='black')
    
    # References
    draw.rectangle([50, 820, 750, 950], fill='lightgray', outline='gray')
    draw.text((70, 840), "References", fill='black')
    draw.text((70, 860), "1. Financial Data Q4 2024", fill='black')
    draw.text((70, 880), "2. Customer Survey Results", fill='black')
    
    return img

def create_form_document():
    """Create a sample form document"""
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Form title
    draw.rectangle([50, 50, 550, 100], fill='darkgreen', outline='green')
    draw.text((200, 70), "EMPLOYMENT APPLICATION", fill='white')
    
    # Personal Information
    draw.rectangle([50, 120, 550, 200], fill='lightblue', outline='blue')
    draw.text((70, 140), "Personal Information", fill='darkblue')
    draw.text((70, 160), "Name: _________________________", fill='black')
    draw.text((70, 180), "Email: _________________________", fill='black')
    
    # Contact Information
    draw.rectangle([50, 220, 550, 300], fill='lightgray', outline='gray')
    draw.text((70, 240), "Contact Information", fill='black')
    draw.text((70, 260), "Phone: _________________________", fill='black')
    draw.text((70, 280), "Address: _______________________", fill='black')
    
    # Education
    draw.rectangle([50, 320, 550, 450], fill='lightyellow', outline='orange')
    draw.text((70, 340), "Education", fill='black')
    draw.text((70, 360), "Degree: _________________________", fill='black')
    draw.text((70, 380), "Institution: ____________________", fill='black')
    draw.text((70, 400), "Graduation Year: _________________", fill='black')
    
    # Experience
    draw.rectangle([50, 470, 550, 600], fill='lightgreen', outline='green')
    draw.text((70, 490), "Work Experience", fill='black')
    draw.text((70, 510), "Previous Company: _______________", fill='black')
    draw.text((70, 530), "Position: _______________________", fill='black')
    draw.text((70, 550), "Duration: _______________________", fill='black')
    
    # Signature
    draw.rectangle([50, 620, 550, 700], fill='lightpink', outline='red')
    draw.text((70, 640), "Signature: ______________________", fill='black')
    draw.text((70, 660), "Date: ___________________________", fill='black')
    
    return img

def create_article_document():
    """Create a sample article document"""
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.rectangle([50, 50, 550, 100], fill='darkred', outline='red')
    draw.text((100, 70), "The Future of AI Technology", fill='white')
    
    # Author and date
    draw.rectangle([50, 110, 550, 140], fill='lightgray', outline='gray')
    draw.text((70, 125), "By: Dr. Jane Smith | January 15, 2024", fill='black')
    
    # Introduction
    draw.rectangle([50, 150, 550, 250], fill='lightblue', outline='blue')
    draw.text((70, 170), "Introduction", fill='darkblue')
    draw.text((70, 190), "Artificial Intelligence is rapidly...", fill='black')
    draw.text((70, 210), "This article explores the latest...", fill='black')
    
    # Main content
    draw.rectangle([50, 270, 550, 500], fill='lightyellow', outline='orange')
    draw.text((70, 290), "Main Content", fill='black')
    draw.text((70, 310), "The development of AI has been...", fill='black')
    draw.text((70, 330), "Machine learning algorithms...", fill='black')
    draw.text((70, 350), "Deep learning networks...", fill='black')
    draw.text((70, 370), "Natural language processing...", fill='black')
    
    # Image/Figure
    draw.rectangle([50, 520, 300, 650], fill='lightgreen', outline='green')
    draw.text((120, 580), "AI Timeline", fill='black')
    
    # Sidebar
    draw.rectangle([320, 520, 550, 650], fill='lightpink', outline='red')
    draw.text((340, 540), "Key Points", fill='black')
    draw.text((340, 560), "• AI growth accelerating", fill='black')
    draw.text((340, 580), "• New applications emerging", fill='black')
    draw.text((340, 600), "• Ethical considerations", fill='black')
    
    # Conclusion
    draw.rectangle([50, 670, 550, 750], fill='lightgray', outline='gray')
    draw.text((70, 690), "Conclusion", fill='black')
    draw.text((70, 710), "The future of AI holds great promise...", fill='black')
    
    return img

def create_receipt_document():
    """Create a sample receipt document"""
    img = Image.new('RGB', (400, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Store header
    draw.rectangle([50, 50, 350, 120], fill='darkblue', outline='blue')
    draw.text((150, 70), "SUPER MART", fill='white')
    draw.text((120, 90), "123 Main Street", fill='white')
    draw.text((140, 110), "City, State 12345", fill='white')
    
    # Receipt details
    draw.rectangle([50, 130, 350, 180], fill='lightgray', outline='gray')
    draw.text((70, 150), "Receipt #: 12345", fill='black')
    draw.text((70, 170), "Date: 2024-01-15", fill='black')
    
    # Items
    items = [
        ("Milk", "$3.50"),
        ("Bread", "$2.25"),
        ("Eggs", "$4.00"),
        ("Cheese", "$5.75")
    ]
    
    y_pos = 200
    for item, price in items:
        draw.rectangle([50, y_pos, 350, y_pos + 30], fill='lightyellow', outline='orange')
        draw.text((70, y_pos + 10), item, fill='black')
        draw.text((300, y_pos + 10), price, fill='black')
        y_pos += 35
    
    # Total
    draw.rectangle([50, y_pos + 10, 350, y_pos + 60], fill='lightgreen', outline='green')
    draw.text((70, y_pos + 30), "Subtotal: $15.50", fill='black')
    draw.text((70, y_pos + 50), "Tax: $1.24", fill='black')
    
    # Grand total
    draw.rectangle([50, y_pos + 70, 350, y_pos + 110], fill='darkgreen', outline='green')
    draw.text((70, y_pos + 90), "Total: $16.74", fill='white')
    
    # Payment info
    draw.rectangle([50, y_pos + 120, 350, y_pos + 160], fill='lightpink', outline='red')
    draw.text((70, y_pos + 140), "Payment: Credit Card", fill='black')
    
    # Footer
    draw.rectangle([50, y_pos + 170, 350, y_pos + 210], fill='lightgray', outline='gray')
    draw.text((120, y_pos + 190), "Thank you for shopping!", fill='black')
    
    return img

def generate_mock_analysis_data():
    """Generate mock analysis data for testing"""
    
    mock_data = {
        "sample_analyses": [
            {
                "filename": "sample_invoice.png",
                "model_used": "layoutparser",
                "layout_blocks": [
                    {"type": "Title", "coordinates": [50, 50, 750, 120], "confidence": 0.98},
                    {"type": "Text", "coordinates": [50, 140, 400, 250], "confidence": 0.95},
                    {"type": "Text", "coordinates": [450, 140, 750, 250], "confidence": 0.92},
                    {"type": "Table", "coordinates": [50, 340, 750, 500], "confidence": 0.89},
                    {"type": "Text", "coordinates": [50, 520, 750, 580], "confidence": 0.94}
                ],
                "ocr_text": "INVOICE INV-2024-001 From: ABC Company 123 Business St City, State 12345 To: XYZ Corp 456 Customer Ave City, State 67890 Invoice Date: 2024-01-15 Due Date: 2024-02-15 Description Quantity Price Total Software License 1 $500.00 $500.00 Support Service 12 $50.00 $600.00 Consulting 40 $100.00 $4000.00 Subtotal: $5100.00 Tax: $510.00 Total: $5610.00 Thank you for your business!",
                "processing_time": 2.34,
                "analysis_date": "2024-01-15T10:30:00Z"
            },
            {
                "filename": "sample_report.png",
                "model_used": "layoutparser",
                "layout_blocks": [
                    {"type": "Title", "coordinates": [50, 50, 750, 120], "confidence": 0.99},
                    {"type": "Text", "coordinates": [50, 140, 750, 220], "confidence": 0.96},
                    {"type": "Text", "coordinates": [50, 240, 750, 400], "confidence": 0.93},
                    {"type": "Figure", "coordinates": [50, 420, 400, 600], "confidence": 0.91},
                    {"type": "Table", "coordinates": [450, 420, 750, 600], "confidence": 0.88},
                    {"type": "Text", "coordinates": [50, 620, 750, 800], "confidence": 0.95},
                    {"type": "Text", "coordinates": [50, 820, 750, 950], "confidence": 0.92}
                ],
                "ocr_text": "QUARTERLY BUSINESS REPORT Abstract This report summarizes our Q4 performance Executive Summary Revenue increased by 15% Customer satisfaction improved New product launches successful Market expansion completed Revenue Chart Key Metrics Revenue: $2.5M Profit: $500K Customers: 1,250 Conclusion The quarter showed strong growth References 1. Financial Data Q4 2024 2. Customer Survey Results",
                "processing_time": 3.12,
                "analysis_date": "2024-01-15T11:15:00Z"
            }
        ],
        "model_performance": {
            "layoutparser": {
                "accuracy": 0.94,
                "avg_processing_time": 2.5,
                "supported_types": ["Text", "Title", "List", "Table", "Figure"]
            },
            "yolo": {
                "accuracy": 0.89,
                "avg_processing_time": 1.8,
                "supported_types": ["Object"]
            }
        },
        "statistics": {
            "total_analyses": 150,
            "avg_blocks_per_document": 6.2,
            "most_common_block_type": "Text",
            "avg_confidence_score": 0.92
        }
    }
    
    # Save mock data
    with open("sample_data/mock_analysis_data.json", "w") as f:
        json.dump(mock_data, f, indent=2)
    
    print("Mock analysis data generated in sample_data/mock_analysis_data.json")

if __name__ == "__main__":
    print("Generating sample data for Document Layout Analysis...")
    
    # Create sample documents
    create_sample_documents()
    
    # Generate mock analysis data
    generate_mock_analysis_data()
    
    print("Sample data generation completed!")
    print("\nGenerated files:")
    print("- sample_data/sample_invoice.png")
    print("- sample_data/sample_report.png")
    print("- sample_data/sample_form.png")
    print("- sample_data/sample_article.png")
    print("- sample_data/sample_receipt.png")
    print("- sample_data/mock_analysis_data.json")
