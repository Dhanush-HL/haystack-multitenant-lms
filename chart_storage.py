"""
Chart Storage System for HayStack Chart Viewer
Handles persistence and retrieval of Chart.js specifications
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional


class ChartStorage:
    """Simple in-memory and file-based chart storage"""
    
    def __init__(self, storage_dir: str = "charts"):
        self.storage_dir = storage_dir
        self.memory_store = {}  # In-memory storage for quick access
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
    
    def store_chart(self, chart_data: Dict[str, Any]) -> str:
        """
        Store chart data and return a unique chart ID
        
        Args:
            chart_data: Dictionary containing chartjs_spec, html_snippet, etc.
            
        Returns:
            Unique chart ID string
        """
        chart_id = f"chart_{uuid.uuid4().hex[:12]}"
        
        # Add metadata
        chart_data['id'] = chart_id
        chart_data['created_at'] = datetime.now().isoformat()
        
        # Store in memory
        self.memory_store[chart_id] = chart_data
        
        # Store in file
        file_path = os.path.join(self.storage_dir, f"{chart_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(chart_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save chart to file: {e}")
        
        return chart_id
    
    def get_chart(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve chart data by ID
        
        Args:
            chart_id: Chart ID to retrieve
            
        Returns:
            Chart data dictionary or None if not found
        """
        # Check memory first
        if chart_id in self.memory_store:
            return self.memory_store[chart_id]
        
        # Check file storage
        file_path = os.path.join(self.storage_dir, f"{chart_id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    chart_data = json.load(f)
                    # Cache in memory for next time
                    self.memory_store[chart_id] = chart_data
                    return chart_data
            except Exception as e:
                print(f"Error loading chart from file: {e}")
        
        return None
    
    def list_charts(self) -> Dict[str, Dict[str, Any]]:
        """
        List all stored charts with basic metadata
        
        Returns:
            Dictionary mapping chart IDs to basic chart info
        """
        charts = {}
        
        # Load from files if not in memory
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    chart_id = filename[:-5]  # Remove .json
                    if chart_id not in self.memory_store:
                        chart_data = self.get_chart(chart_id)
                        if chart_data:
                            charts[chart_id] = {
                                'title': chart_data.get('title', 'Untitled Chart'),
                                'chart_type': chart_data.get('chart_type', 'unknown'),
                                'created_at': chart_data.get('created_at', 'unknown')
                            }
        
        # Add from memory
        for chart_id, chart_data in self.memory_store.items():
            charts[chart_id] = {
                'title': chart_data.get('title', 'Untitled Chart'),
                'chart_type': chart_data.get('chart_type', 'unknown'),
                'created_at': chart_data.get('created_at', 'unknown')
            }
        
        return charts
    
    def delete_chart(self, chart_id: str) -> bool:
        """
        Delete a chart by ID
        
        Args:
            chart_id: Chart ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        # Remove from memory
        if chart_id in self.memory_store:
            del self.memory_store[chart_id]
            deleted = True
        
        # Remove file
        file_path = os.path.join(self.storage_dir, f"{chart_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted = True
            except Exception as e:
                print(f"Error deleting chart file: {e}")
        
        return deleted


# Global chart storage instance
chart_storage = ChartStorage()