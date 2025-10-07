#!/usr/bin/env python3
"""
Simple web server to serve the chart viewer HTML file
"""

import http.server
import socketserver
import webbrowser
import os
import time
import threading

def start_chart_viewer():
    """Start a simple web server for the chart viewer"""
    PORT = 8080
    
    # Change to the directory containing the HTML file
    os.chdir(r'C:\Users\Dhanush-HL\OneDrive - Human Logic\Documents\HayStack\Haystack')
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Chart Viewer server started at http://localhost:{PORT}")
            print(f"📊 Opening chart viewer in your browser...")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(2)
                webbrowser.open(f'http://localhost:{PORT}/chart_viewer.html')
            
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            print("🔧 Make sure your HayStack FastAPI server is running on port 8000")
            print("⚡ Press Ctrl+C to stop the server")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Chart Viewer server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_chart_viewer()