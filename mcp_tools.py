"""
MCP-backed Haystack Tools
Provides DBQueryTool and ChartTool that interface with existing MCP server
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from haystack import component
from dataclasses import dataclass
import os


@dataclass
class DBQueryResult:
    """Result from DB query tool"""
    content: str
    sql: str
    rows: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass  
class ChartResult:
    """Result from Chart tool"""
    chartjs_spec: Dict[str, Any]
    chart_view_url: str
    sql_used: str
    title: str
    error: Optional[str] = None


class MCPClient:
    """Client for communicating with MCP server"""
    
    def __init__(self, mcp_url: str = "http://localhost:3001"):
        self.mcp_url = mcp_url
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via HTTP"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "tool": tool_name,
                    "arguments": arguments
                }
                
                async with session.post(f"{self.mcp_url}/mcp/tool", json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {
                            "error": f"MCP call failed with status {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {"error": f"MCP client error: {str(e)}"}


@component
class DBQueryTool:
    """
    Haystack Component that calls MCP haystack_query for safe SQL generation and execution
    """
    
    def __init__(self, mcp_url: str = "http://localhost:3001"):
        self.mcp_client = MCPClient(mcp_url)
        
    @component.output_types(result=DBQueryResult)
    def run(self, question: str, user_id: int) -> Dict[str, Any]:
        """
        Execute a database query via MCP server
        
        Args:
            question: Natural language question about the data
            user_id: ID of the requesting user (for RBAC)
            
        Returns:
            Dictionary with DBQueryResult
        """
        try:
            # Call MCP haystack_query tool
            loop = asyncio.get_event_loop()
            mcp_result = loop.run_until_complete(
                self.mcp_client.call_tool("haystack_query", {
                    "query": question,
                    "user_id": user_id
                })
            )
            
            if "error" in mcp_result:
                return {"result": DBQueryResult(
                    content="",
                    sql="",
                    rows=[],
                    error=mcp_result["error"]
                )}
            
            # Parse MCP response
            content = mcp_result.get("content", "")
            sql = mcp_result.get("sql", "")
            rows = mcp_result.get("rows", [])
            
            # Format the content for Haystack Agent
            formatted_content = f"""Query Results:
{content}

SQL Used: {sql}
Rows Returned: {len(rows)}"""
            
            result = DBQueryResult(
                content=formatted_content,
                sql=sql,
                rows=rows
            )
            
            return {"result": result}
            
        except Exception as e:
            return {"result": DBQueryResult(
                content="",
                sql="",
                rows=[],
                error=f"DBQueryTool error: {str(e)}"
            )}


@component
class ChartTool:
    """
    Haystack Component that calls MCP haystack_chart for chart generation
    """
    
    def __init__(self, mcp_url: str = "http://localhost:3001"):
        self.mcp_client = MCPClient(mcp_url)
        
    @component.output_types(result=ChartResult)  
    def run(self, question: str, user_id: int) -> Dict[str, Any]:
        """
        Generate a chart via MCP server
        
        Args:
            question: Natural language question for chart generation
            user_id: ID of the requesting user (for RBAC)
            
        Returns:
            Dictionary with ChartResult
        """
        try:
            # Call MCP haystack_chart tool
            loop = asyncio.get_event_loop()
            mcp_result = loop.run_until_complete(
                self.mcp_client.call_tool("haystack_chart", {
                    "query": question,
                    "user_id": user_id
                })
            )
            
            if "error" in mcp_result:
                return {"result": ChartResult(
                    chartjs_spec={},
                    chart_view_url="",
                    sql_used="",
                    title="Error",
                    error=mcp_result["error"]
                )}
            
            # Parse MCP response
            chartjs_spec = mcp_result.get("chartjs_spec", {})
            chart_view_url = mcp_result.get("chart_view_url", "")
            sql_used = mcp_result.get("sql", "")
            title = mcp_result.get("title", "Chart")
            
            result = ChartResult(
                chartjs_spec=chartjs_spec,
                chart_view_url=chart_view_url,
                sql_used=sql_used,
                title=title
            )
            
            return {"result": result}
            
        except Exception as e:
            return {"result": ChartResult(
                chartjs_spec={},
                chart_view_url="",
                sql_used="",
                title="Error",
                error=f"ChartTool error: {str(e)}"
            )}


# Tool factory functions for easy registration with Haystack Agent
def create_db_query_tool(mcp_url: str = "http://localhost:3001") -> DBQueryTool:
    """Factory function to create DBQueryTool"""
    return DBQueryTool(mcp_url=mcp_url)


def create_chart_tool(mcp_url: str = "http://localhost:3001") -> ChartTool:
    """Factory function to create ChartTool"""  
    return ChartTool(mcp_url=mcp_url)


# Utility functions for Agent integration
def format_db_result_for_agent(result: DBQueryResult) -> str:
    """Format DB query result for Agent response"""
    if result.error:
        return f"❌ Database Query Error: {result.error}"
    
    response = f"""📊 **Query Results**

{result.content}

🔍 **SQL Query Used:**
```sql
{result.sql}
```

📈 **Data Summary:** {len(result.rows)} rows returned"""
    
    return response


def format_chart_result_for_agent(result: ChartResult) -> str:
    """Format chart result for Agent response"""
    if result.error:
        return f"❌ Chart Generation Error: {result.error}"
    
    chart_type = result.chartjs_spec.get("type", "unknown")
    data_points = 0
    if "data" in result.chartjs_spec and "datasets" in result.chartjs_spec["data"]:
        for dataset in result.chartjs_spec["data"]["datasets"]:
            if "data" in dataset:
                data_points += len(dataset["data"])
    
    response = f"""📈 **Chart Generated: {result.title}**

**Chart Type:** {chart_type.title()}
**Data Points:** {data_points}

**View Chart:** [{result.chart_view_url}]({result.chart_view_url})

🔍 **SQL Query Used:**
```sql  
{result.sql_used}
```

The chart has been generated and stored. You can view it using the link above."""
    
    return response