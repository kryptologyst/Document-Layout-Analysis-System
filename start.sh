#!/bin/bash

# Document Layout Analysis System Startup Script
# This script sets up and runs the complete system

set -e

echo "🚀 Document Layout Analysis System"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Generate sample data
generate_sample_data() {
    print_status "Generating sample data..."
    python generate_sample_data.py
    print_success "Sample data generated"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data logs sample_data
    print_success "Directories created"
}

# Check if sample data exists
check_sample_data() {
    if [ ! -d "sample_data" ] || [ -z "$(ls -A sample_data)" ]; then
        print_warning "Sample data not found. Generating..."
        generate_sample_data
    else
        print_success "Sample data found"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if python -m pytest test_suite.py -v; then
        print_success "All tests passed"
    else
        print_warning "Some tests failed, but continuing..."
    fi
}

# Start the system
start_system() {
    print_status "Starting Document Layout Analysis System..."
    echo ""
    echo "🌐 Web Interface: http://localhost:8501"
    echo "🔌 API Server: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo ""
    echo "Press Ctrl+C to stop the system"
    echo ""
    
    # Start API server in background
    python api.py &
    API_PID=$!
    
    # Wait a moment for API to start
    sleep 3
    
    # Start Streamlit app
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    
    # Cleanup on exit
    trap "kill $API_PID 2>/dev/null || true" EXIT
}

# Main function
main() {
    echo "Starting setup process..."
    echo ""
    
    # Check prerequisites
    check_python
    check_pip
    
    # Setup environment
    create_venv
    activate_venv
    install_dependencies
    
    # Setup data and directories
    create_directories
    check_sample_data
    
    # Optional: Run tests
    if [ "$1" = "--test" ]; then
        run_tests
    fi
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    
    # Start the system
    start_system
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Document Layout Analysis System Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --test     Run tests before starting"
        echo "  --help     Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                # Start the system"
        echo "  $0 --test         # Run tests and start the system"
        echo ""
        exit 0
        ;;
    --test)
        main --test
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
